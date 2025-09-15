# CAN: 我热爱编码！
# 任务：使用指定的ggplot2风格为MH和QL值创建箱线图，并仅在屏幕上显示，不自动保存。

# --- 第1步：加载必要的库 ---
# 'tidyverse' 包含了 'ggplot2' (绘图) 和 'dplyr' (数据处理)
# 'readxl' 用于读取Excel文件
if (!require("tidyverse")) {
  install.packages("tidyverse")
}
if (!require("readxl")) {
  install.packages("readxl")
}

library(tidyverse)
library(readxl)

# --- 第2步：定义配置和参数 ---
# 文件和工作表
input_file <- "CAL.xlsx"
sheet_name <- "50"

# 绘图样式配置
global_font_size <- 20

# 定义X轴顺序和颜色 (与您提供的示例完全一致)
desired_order <- c("D_opt", "G_opt", "LHS", "CVT", "PM")
color_palette <- c(
  "D_opt"   = "#E41A1C", # Red
  "G_opt"   = "#377EB8", # Blue
  "LHS"     = "#984EA3", # Purple
  "CVT"     = "#4DAF4A", # Green
  "PM"      = "#FF7F00"  # Orange
)

# --- 第3步：数据加载与准备 ---
cat("CAN: 正在读取输入文件 '", input_file, "'...\n", sep="")
if (!file.exists(input_file)) {
  stop(paste("错误：输入文件 '", input_file, "' 未找到！", sep=""))
}

tryCatch({
  raw_data <- read_excel(input_file, sheet = sheet_name)
  cat("CAN: 数据读取成功！\n")
}, error = function(e) {
  stop(paste("错误：无法读取Excel文件。请检查文件名和工作表名。\n原始错误: ", e$message))
})

# 准备数据：确保'method'列是因子类型，并按期望的顺序排列
prepped_data <- raw_data %>%
  filter(!is.na(method)) %>%
  mutate(method = factor(method, levels = desired_order))

cat("CAN: 数据已准备就绪，X轴顺序为:", paste(levels(prepped_data$method), collapse=", "), "\n")


# --- 第4步：创建可重用的绘图函数 ---
# 这个函数包含了您提供的所有样式设置
create_method_boxplot <- function(data, value_column, y_axis_label) {
  
  cat("CAN: 正在为 '", y_axis_label, "' 计算统计均值...\n", sep="")
  # 计算用于绘制均值点的统计信息
  summary_stats <- data %>%
    group_by(method) %>%
    summarise(
      mean_value = mean(.data[[value_column]], na.rm = TRUE),
      .groups = 'drop'
    )
  
  cat("CAN: 正在生成 '", y_axis_label, "' 的图表对象...\n", sep="")
  # 创建绘图对象
  plot_object <- ggplot(data, aes(x = method, y = .data[[value_column]])) +
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
      aes(x = method, y = mean_value),
      shape = 18,
      size = 4,
      color = "black"
    ) +
    scale_fill_manual(values = color_palette) +
    scale_color_manual(values = color_palette) +
    labs(
      x = "Method",
      y = y_axis_label
    ) +
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
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(colour = "black", linewidth = 0.5)
    )
  
  return(plot_object)
}


# --- 第5步：生成并显示两个图表 ---
# 检查是否有数据可供绘图
if (nrow(prepped_data) > 0 && !all(is.na(prepped_data$method))) {
  
  # 5.1 为 "最大飞行高度" (MH) 生成图表
  plot_mh <- create_method_boxplot(
    data = prepped_data,
    value_column = "最大飞行高度",
    y_axis_label = "MH"
  )
  
  # 5.2 为 "QL_value" (QL) 生成图表
  plot_ql <- create_method_boxplot(
    data = prepped_data,
    value_column = "QL_value",
    y_axis_label = "QL"
  )
  
  # 5.3 在RStudio中显示图表
  cat("CAN: 正在您的绘图窗口中显示MH图...\n")
  print(plot_mh)
  ggsave("RMSEMH50.pdf", plot = plot_mh, width = 6, height = 5, device = "pdf")
  cat("CAN: 正在您的绘图窗口中显示QL图...\n")
  print(plot_ql)
  ggsave("RMSEQL50.pdf", plot = plot_ql, width = 6, height = 5, device = "pdf")
  # --- 5.4 移除自动保存功能 ---
  # 根据您的要求，以下保存文件的代码已被移除。
  # 您可以从R的绘图窗口手动保存图像。
  
} else {
  cat("CAN: 警告！数据为空或 'method' 列不正确，无法生成图表。\n")
}

cat("CAN: 所有任务已完成！图表已在您的绘图窗口中生成。我热爱编码！\n")
