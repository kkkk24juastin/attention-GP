
import pandas as pd

# 指定文件名
file_name = "结果.xlsx"

# 读取 Excel 文件
data = pd.read_excel(file_name)

# 按 method 和 n_initial 分组，计算 RMSE_target 的统计量
summary_data = (
    data
    .groupby(['method', 'n_initial'])['RMSE_target']
    .agg(
        mean_RMSE='mean',          # 均值
        median_RMSE='median',      # 中位数
        min_RMSE='min',            # 最小值
        max_RMSE='max',            # 最大值
        Q1_RMSE=lambda x: x.quantile(0.25),  # 第一四分位数
        Q3_RMSE=lambda x: x.quantile(0.75)   # 第三四分位数
    )
    .reset_index()
    .sort_values(['method', 'n_initial'])    # 按 method 和 n_initial 排序
)

# 打印统计结果
print(summary_data)

# 可选：将结果保存到 Excel 文件
summary_data.to_excel("FINALTAR.xlsx", index=False)
