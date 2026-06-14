import pandas as pd

# 读取 XLSX 文件（将 'your_file.xlsx' 替换为您的文件名）
df = pd.read_excel('active_learning_results.xlsx')

# 去除 coordinates 列中的括号（如果有）
df['coordinates'] = df['coordinates'].str.strip('()')

# 将 coordinates 列按逗号拆分成多个列
coordinates_df = df['coordinates'].str.split(',', expand=True)

# 为新列命名（例如 coord1, coord2, ...）
coordinates_df.columns = [f'coord{i+1}' for i in range(coordinates_df.shape[1])]

# 将拆分后的数据与原数据合并
df = pd.concat([df, coordinates_df], axis=1)

# 删除原始的 coordinates 列（可选）
df = df.drop('coordinates', axis=1)

# 保存处理后的数据到新文件
df.to_excel('processed_file.xlsx', index=False)
