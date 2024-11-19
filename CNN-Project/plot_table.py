import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 创建DataFrame
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [24, 30, 18],
    "Job": ["Engineer", "Scientist", "Artist"]
})

# 使用seaborn绘制表格
sns.set(style="whitegrid")
# 使用关键字参数调用pivot方法
pivot_df = df.pivot(index="Job", columns="Name", values="Age")
ax = sns.heatmap(pivot_df, annot=True, fmt=".2f", linewidths=.5, square=True)

# 显示表格
plt.show()