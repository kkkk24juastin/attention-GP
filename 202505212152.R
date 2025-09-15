# 加载必要的包
library(lhs)
library(writexl)


# 生成LHS样本（50个样本，5个变量）
lhs_samples <- randomLHS(20, 5)

# 缩放样本到指定范围
head_cone_length <- lhs_samples[,1] * 20  # 头锥长度：0-20
base_diameter_ratio <- lhs_samples[,2] * 3 + 2  # 头锥底座直径/箭体外直径：2-5
wall_thickness <- lhs_samples[,3] * 1  # 头锥壁厚：0-1
body_length <- lhs_samples[,4] * 20 + 30  # 箭体长度：30-50
inner_diameter <- lhs_samples[,5] * 1 + 1  # 箭体内直径：1-2

# 创建数据框
data <- data.frame(
  head_cone_length = head_cone_length,
  base_diameter_ratio = base_diameter_ratio,
  wall_thickness = wall_thickness,
  body_length = body_length,
  inner_diameter = inner_diameter
)

# 四舍五入到小数点后二位（仅对数值型变量）
data[, c(1,4)] <- round(data[, c(1,4)], 1)
data[, c(2,3,5)] <- round(data[, c(2,3,5)], 2)

# 根据头锥长度添加分类列
data$category <- ifelse(data$head_cone_length <= 10, 0, 1)

# 设置中文列名
colnames(data) <- c("头锥长度", "头锥底座直径/箭体外直径", "头锥壁厚", "箭体长度", "箭体内直径", "头锥长度分类")

# 保存为xlsx文件
write_xlsx(data, "samples.xlsx")

# 验证数据（可选）
summary(data)
table(data$`头锥长度分类`)
