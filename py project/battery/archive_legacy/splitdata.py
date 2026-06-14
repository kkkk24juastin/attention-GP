import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('data.xlsx')

# 根据 tem 列分组
grouped = df.groupby('tem')

# 初始化训练集和池集的列表
train_dfs = []
pool_dfs = []

# 对每个 tem 组进行处理
for name, group in grouped:
    if len(group) >= 60:
        # 随机选择 60 条数据作为训练集
        train = group.sample(n=70, random_state=42)
        # 剩余数据作为池集
        pool = group.drop(train.index)
    else:
        # 如果数据少于 60 条，全部放入训练集
        train = group
        # 池集为空
        pool = pd.DataFrame(columns=group.columns)

    # 将训练集和池集添加到列表中
    train_dfs.append(train)
    pool_dfs.append(pool)

# 合并所有训练集和池集
train_df = pd.concat(train_dfs)
pool_df = pd.concat(pool_dfs)

# 保存到 Excel 文件
train_df.to_excel('train.xlsx', index=False)
pool_df.to_excel('pool.xlsx', index=False)

print("处理完成！训练集已保存至 train.xlsx，剩余数据已保存至 pool.xlsx。")
