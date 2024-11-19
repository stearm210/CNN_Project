import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df_srresnet_mlka = pd.read_csv('results_csv/SRResNet(MLKA)_MSE.csv')
df_srresnet = pd.read_csv('results_csv/SRResNet_MSE.csv')

# 清洗数据，确保Step和Value列是正确的数据类型
df_srresnet_mlka['Step'] = df_srresnet_mlka['Step'].astype(int)
df_srresnet_mlka['Value'] = df_srresnet_mlka['Value'].astype(float)
df_srresnet['Step'] = df_srresnet['Step'].astype(int)
df_srresnet['Value'] = df_srresnet['Value'].astype(float)

# 绘制折线图
plt.figure(figsize=(12, 6))

# 绘制SRResNet(MLKA)的MSE损失
plt.plot(df_srresnet_mlka['Step'], df_srresnet_mlka['Value'], label='SRResNet(MLKA)', marker='o')

# 绘制SRResNet的MSE损失
plt.plot(df_srresnet['Step'], df_srresnet['Value'], label='SRResNet', marker='x')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('MSE Loss Over Training Steps')
plt.xlabel('Training Step')
plt.ylabel('MSE Loss')

# 显示图表
plt.show()