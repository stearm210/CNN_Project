import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义的层归一化类，用于对数据进行归一化处理
"""
forward 方法根据 data_format 参数（'channels_last' 或 'channels_first'）来决定如何对输入数据进行归一化。
对于 'channels_last'，直接使用 PyTorch 的内置 F.layer_norm 函数；对于 'channels_first'，则手动计算均值和方差，并进行归一化。
"""
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized = nn.LayerNorm(normalized_shape,)


    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


"""
首先，输入特征通过 LayerNorm 进行归一化。
然后，输入特征被分为两部分，一部分通过 Conv1 进行卷积操作，另一部分通过 DWConv1（深度可分离卷积）进行空间注意力计算。
空间注意力的结果与另一部分特征相乘，增强了重要的空间信息。
最后，通过 Conv2 和 scale 参数进行最后的调整。
"""
# 门控空间注意力单元（GSAU）：一个注意力机制模块，通过空间注意力来增强特征图中的重要信息。
class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

"""
输入特征首先通过 LayerNorm 进行归一化。
然后，输入特征被分割成三部分，每部分通过不同尺度的卷积核（LKA7、LKA5、LKA3）进行处理。
每个尺度的特征都通过深度可分离卷积和膨胀卷积来提取特征，并通过 X5、X7 进行进一步的调整。
最后，将不同尺度的特征合并，并与原始特征相乘，通过 proj_last 和 scale 进行最后的调整。
"""
# 多尺度大核注意模块（MLKA）：通过不同大小的卷积核来捕获不同尺度的特征，增强模型对多尺度信息的捕捉能力
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            n_feats = (n_feats // 3 + 1) * 3
            raise ValueError(f'Warning: n_feats adjusted to {n_feats} to be divisible by 3.')

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3,1,1,0)
        )
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)
        )
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)
        )

        self.x3 = nn.Conv2d(n_feats // 3, n_feats // 3,3,1,1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups = n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups = n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        )

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        )

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X5(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)], dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x



# 多尺度注意力块（MAB）：MAB是一个组合模块，它结合了MLKA和GSAU的功能，旨在同时利用多尺度注意力和空间注意力来增强特征
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)

    def forward(self, x):
        x = self.LKA(x)
        x = self.LFE(x)
        return x
