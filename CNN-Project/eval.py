import torch.utils.data
from utils import *
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
from model import SRResNet, SRResNet_MLKA
import time

# 模型参数
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
scaling_factor = 4
ngpu = 1
device = torch.device('cuda')

if __name__ == '__main__':
    # 测试集目录
    data_folder = "./data/"
    test_data_names = ['Set5', "Set14", "BSD100"]

    # 预训练模型
    srresnet_checkpoint = "results/checkpoint_srresnet.pth"

    # 加载模型
    checkpoint = torch.load(srresnet_checkpoint)
    srresnet = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size, n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
    srresnet = srresnet.to(device)
    srresnet.load_state_dict(checkpoint['model'])

    # 多GPU测试
    if torch.cuda.is_available() and ngpu > 1:
        srresnet = nn.DataParallel(srresnet, device_ids=list(range(ngpu)))

    srresnet.eval()
    model = srresnet

    for test_data_name in test_data_names:
        print("\n数据集 %s:\n" % test_data_name)

        # 定制化数据加载器
        test_dataset = SRDataset(data_folder, split='test', crop_size=0, scaling_factor=4, lr_img_type='imagenet-norm', hr_img_type='[-1, 1]', test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        # 记录每个样本PSNR和SSIM值
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # 记录测试时间
        start = time.time()

        with torch.no_grad():
            # 逐批样本进行推理计算
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # 数据移至默认设备
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # 前向传播
                sr_imgs = model(lr_imgs)

                # 计算 PSNR 和 SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)

                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

        # 输出平均PSNR和SSIM
        print('PSNR {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM {ssims.avg:.3f}'.format(ssims=SSIMs))
        print('平均单张样本用时 {:.3f} 秒'.format((time.time() - start)/len(test_dataset)))

    print('\n')