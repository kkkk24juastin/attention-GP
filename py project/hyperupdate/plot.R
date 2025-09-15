# CAN: "立即编写任何代码" R 脚本 (灵活、完整、无标签版)
# 我爱编程。

# --------------------------------------------------------------------------
# 步骤 0: 用户配置 - 在这里修改！
# --------------------------------------------------------------------------
# --- 1. 定义要分析的超参数列名 ---
# 在下面的向量中，放入您希望从 Excel 文件中绘制的列名。
# 您可以只放一个，例如 c("nug")
# 您也可以放多个，例如 c("nug", "d1", "d2", "g")
# lambda	beta0	beta1	beta2	nug	d1	d2
parameters_to_plot <- c("beta0","d2")

# --- 2. 设置文件路径 ---
# 确保 Excel 文件位于您的 R 工作目录中，或提供完整路径
file_path <- "my_btgp_trace1.xlsx"

# --- 3. 设置输出图像的文件名 ---
output_filename <- "diagnostic_plots_no_labels.png"


# --------------------------------------------------------------------------
# 步骤 1: 安装并加载必要的 R 包
# --------------------------------------------------------------------------
# 确保所有需要的包都已安装
# 如果您尚未安装这些包，请取消下面这行代码的注释并运行它
# install.packages(c("readxl", "ggplot2", "cowplot"))

# 加载 R 包
library(readxl)
library(ggplot2)
library(cowplot)

cat("R 包已成功加载。\n")

# --------------------------------------------------------------------------
# 步骤 2: 读取和准备数据
# --------------------------------------------------------------------------
# 检查文件是否存在
if (!file.exists(file_path)) {
  stop("错误: 文件 '", file_path, "' 不存在。请检查 file_path 变量。")
}

# 读取 Excel 文件
data <- read_excel(file_path)
cat("数据已成功从 '", file_path, "' 读取，共 ", nrow(data), " 行和 ", ncol(data), " 列。\n")

# 为轨迹图添加一个迭代索引列
data$index <- 1:nrow(data)

# 验证所选参数是否存在于数据中
valid_params <- character(0)
for (param in parameters_to_plot) {
  if (param %in% names(data)) {
    valid_params <- c(valid_params, param)
  } else {
    warning("警告: 参数 '", param, "' 在 Excel 文件中未找到，将被忽略。")
  }
}

# 如果没有一个有效参数，则停止脚本
if (length(valid_params) == 0) {
  stop("错误: 配置中指定的所有参数都不在数据文件中。脚本无法继续。")
}
parameters_to_plot <- valid_params # 只使用有效的参数继续


# --------------------------------------------------------------------------
# 步骤 3: 定义一个通用的绘图函数
# --------------------------------------------------------------------------
# 这个函数为任何给定的参数名称生成一组诊断图。
create_diagnostic_plot <- function(df, param_name) {
  
  cat("正在为参数 '", param_name, "' 创建诊断图...\n")
  
  # (a-1) ACF (自相关函数) 图
  acf_obj <- acf(df[[param_name]], plot = FALSE, lag.max = 10000)
  acf_df <- data.frame(lag = acf_obj$lag, acf = acf_obj$acf)
  conf_limit <- qnorm((1 + 0.95) / 2) / sqrt(acf_obj$n.used)
  
  plot_acf <- ggplot(acf_df, aes(x = .data$lag, y = .data$acf)) +
    geom_hline(yintercept = 0, color = "grey") +
    geom_segment(aes(xend = .data$lag, yend = 0), color = "#00008B") +
    geom_hline(yintercept = c(conf_limit, -conf_limit), linetype = "dashed", color = "blue") +
    ylim(-1, 1) +
    labs(
      x = "Lag Length",
      y = paste("ACF for", param_name)
    ) +
    theme_bw() +
    theme(plot.margin = margin(t = 20, r = 5.5, b = 5.5, l = 5.5, unit = "pt"))
  
  # (a-2) 轨迹图 (Trace Plot)
  plot_trace <- ggplot(df, aes(x = .data$index, y = .data[[param_name]])) +
    geom_line(color = "#00008B") +
    labs(x = "Iteration", y = param_name) +
    theme_bw()
  
  # (a-3) 直方图 (Histogram)
  plot_hist <- ggplot(df, aes(x = .data[[param_name]])) +
    geom_histogram(fill = "#00008B", color = "white", bins = 20) +
    coord_flip() +
    labs(x = "", y = "") +
    theme_bw() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    )
  
  # 组合底部图 (直方图 + 轨迹图)
  bottom_row <- plot_grid(
    plot_hist, plot_trace,
    ncol = 2,
    align = 'h',
    axis = 'tb',
    rel_widths = c(0.4, 1)
  )
  
  # 组合该参数的所有图
  plot_group <- plot_grid(
    plot_acf,
    bottom_row,
    nrow = 2,
    rel_heights = c(1, 1)
  )
  
  return(plot_group)
}


# --------------------------------------------------------------------------
# 步骤 4: 循环生成所有请求的图
# --------------------------------------------------------------------------
# 创建一个空列表来存储每个参数的组合图
plot_list <- list()

# 遍历在步骤 0 中定义好的参数列表
for (param in parameters_to_plot) {
  plot_list[[param]] <- create_diagnostic_plot(data, param)
}

cat("所有单个参数的图表已生成。\n")

# --------------------------------------------------------------------------
# 步骤 5: 组合并保存最终图像
# --------------------------------------------------------------------------
# *** 修改点：下面的 `labels` 参数已被移除 ***
final_plot <- plot_grid(
  plotlist = plot_list,
  ncol = length(plot_list) # 将所有图并排排列
)

# 在 RStudio 的绘图窗口中显示最终图像
cat("正在显示最终的组合图像...\n")
print(final_plot)

# 将图像保存为文件
ggsave(
  output_filename,
  final_plot,
  width = 6 * length(plot_list),
  height = 8,
  dpi = 300,
  limitsize = FALSE
)

cat("任务完成！最终图像已保存为 '", output_filename, "'。\n")

# --- 脚本结束 ---
