# --- 加载必要的库 ---
library(ggplot2)
library(readxl)
library(dplyr)
# RColorBrewer 仍然可能被其他部分间接使用，或者作为备用，保留是安全的
if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
  install.packages("RColorBrewer")
}
library(RColorBrewer)

# --- 1. 全局字体大小控制 ---
global_font_size <- 20

# --- 2. 数据加载与准备 ---
excel_file_path <- "结果.xlsx" # 确保文件名或路径正确
sheet_name <- 1 # 假设数据在第一个工作表

# 检查文件是否存在
if (!file.exists(excel_file_path)) {
  stop(paste("错误: Excel文件 '", excel_file_path, "' 未找到。请确保文件存在于工作目录中或提供正确路径。", sep=""))
}

# 读取原始数据
raw_data <- read_excel(excel_file_path, sheet = sheet_name)
cat("CAN: 已成功从 '", excel_file_path, "' 读取数据。\n", sep="")

# --- 3. 数据清洗与预处理 ---
rmse_column_to_use <- "RMSE_target" # 您可以根据需要更改此列名

# 检查必要的列是否存在
if (!("method" %in% colnames(raw_data))) {
  stop(paste("错误: Excel文件中未找到 'method' 列。"))
}
if (!(rmse_column_to_use %in% colnames(raw_data))) {
  stop(paste("错误: Excel文件中未找到 '", rmse_column_to_use, "' 列。", sep=""))
}

# 重命名 'K_means' (如果存在) 并过滤NA值
raw_data <- raw_data %>%
  mutate(method = case_when(
    method == "K_means" ~ "CVT",
    TRUE ~ as.character(method)
  )) %>%
  filter(!is.na(.data[[rmse_column_to_use]]))

# --- 【本次核心修改】: 定义X轴顺序和颜色方案 ---

# 1. 定义期望的X轴顺序
desired_order <- c("D_opt", "G_opt", "LHS", "CVT", "PM")
cat("CAN: 将使用以下X轴顺序:", paste(desired_order, collapse=", "), "\n")

# 2. 将'method'列转换为因子，并使用'desired_order'指定水平顺序
# 注意：Excel中没有的method会被忽略，不会报错
available_methods <- intersect(desired_order, unique(raw_data$method))
raw_data$method <- factor(raw_data$method, levels = available_methods)
raw_data <- raw_data %>% filter(!is.na(method)) # 过滤掉不在 desired_order 中的方法

method_names <- levels(raw_data$method)
n_methods <- length(method_names)

# --- 4. 计算摘要统计信息 (用于绘制均值点) ---
summary_stats <- raw_data %>%
  group_by(method) %>%
  summarise(
    mean_RMSE = mean(.data[[rmse_column_to_use]], na.rm = TRUE),
    .groups = 'drop'
  )

# --- 5. 【本次核心修改】: 使用您指定的统一颜色调色板 ---

# 定义颜色方案
recommended_colors <- c(
  "D_opt"   = "#E41A1C", # Red
  "G_opt"   = "#377EB8", # Blue
  "CVT"     = "#4DAF4A", # Green
  "LHS"     = "#984EA3", # Purple
  "PM"      = "#FF7F00"  # Orange
)

# 从推荐颜色中选取当前数据中存在的那些，并按正确的顺序排列
final_colors <- recommended_colors[method_names]
names(final_colors) <- method_names
cat("CAN: 已采用您指定的统一调色板。\n")

# --- 6. 创建箱线图并叠加散点 ---
rmse_boxplot_final <- ggplot(raw_data, aes(x = method, y = .data[[rmse_column_to_use]])) +
  
  geom_jitter(
    aes(color = method),
    width = 0.25,
    size = 2.0,
    alpha = 0.9
  ) +
  
  geom_boxplot(
    aes(fill = method),
    width = 0.6,
    lwd = 0.8,
    alpha = 0.7,
    outlier.shape = NA
  ) +
  
  geom_point(
    data = summary_stats,
    aes(x=method, y = mean_RMSE),
    shape = 18,        # 菱形
    size = 4,
    color = "black"
  ) +
  
  # --- 7. 【本次核心修改】: 自定义外观以使用统一颜色和最佳主题 ---
  scale_fill_manual(
    values = final_colors
  ) +
  
  scale_color_manual(
    values = final_colors
  ) +
  
  labs(
    x = "Method",
    y = "RMSE"
  ) +
  
  # 改用 theme_bw() 作为基础，以获得更好的坐标轴线控制
  theme_bw(base_size = global_font_size) +
  
  theme(
    plot.title = element_blank(),
    axis.text = element_text(face = "bold", color = "black"),
    axis.title.x = element_text(face = "bold", margin = margin(t = 15)),
    axis.title.y = element_text(
      face = "bold",
      angle = 0,
      vjust = 0.5,
      margin = margin(r = 15)
    ),
    legend.position = "none", # 隐藏图例
    
    # 移除网格线和边框，只保留坐标轴线
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(colour = "black", linewidth = 0.5)
  )

# --- 8. 显示图表 ---
if (n_methods > 0) {
  print(rmse_boxplot_final)
  cat("CAN: 代码执行完毕。已根据您的统一色系要求，从Excel文件生成了最终图表。\n")
} else {
  cat("CAN: 警告！在过滤后，没有足够的数据来生成图表。\n")
}
ggsave("RMSE5d.pdf", plot = rmse_boxplot_final, width = 6, height = 5, device = "pdf")
# --- 9. (可选) 保存图表 ---
# if (n_methods > 0) {
#   ggsave(
#     "RMSE_boxplot_from_Excel_final.png",
#     plot = rmse_boxplot_final,
#     width = 8, 
#     height = 6,
#     dpi = 300
#   )
#   cat("CAN: 图像已保存为 'RMSE_boxplot_from_Excel_final.png'\n")
# }
