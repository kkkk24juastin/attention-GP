# CAN: R Script v6.0 - Defined Sheet Processing
# 我热爱编码！

# --- 第1步：检查并安装所需的程序包 ---
# 我们需要 'tidyverse', 'readxl', 和 'openxlsx'。
cat("CAN: 正在检查并准备所需的工具包...\n")
if (!require("tidyverse")) {
  install.packages("tidyverse")
}
if (!require("readxl")) {
  install.packages("readxl")
}
if (!require("openxlsx")) {
  install.packages("openxlsx")
}

# --- 第2步：加载程序包 ---
library(tidyverse)
library(readxl)
library(openxlsx)

# --- 第3步：用户配置区域 ---
# --- !!! 重要：请在这里修改您要处理的工作表名称 !!! ---
sheet_to_process <- "80" 
# --- 例如，下次运行时，您可以将其修改为: sheet_to_process <- "60" ---

# 定义输入和输出文件名
input_file <- "CAL.xlsx"
output_file <- "COMPARE.xlsx"

# --- 第4步：检查和验证 ---
# 4.1: 检查输入文件是否存在
if (!file.exists(input_file)) {
  stop(paste("错误：输入文件 '", input_file, "' 未找到！请确保它和R脚本在同一个文件夹下。", sep = ""))
}

# 4.2: 验证您指定的工作表是否存在于输入文件中 (这是一个好的实践)
cat("CAN: 正在验证工作表 '", sheet_to_process, "' 是否存在于 '", input_file, "'...\n", sep="")
available_sheets <- tryCatch({
  excel_sheets(input_file)
}, error = function(e) {
  stop(paste("错误：无法读取 '", input_file, "' 的工作表列表。请检查文件是否损坏或格式是否正确。", sep=""))
})

if (!(sheet_to_process %in% available_sheets)) {
  stop(paste("错误：在 '", input_file, "' 中未找到名为 '", sheet_to_process, "' 的工作表！\n",
             "可用工作表包括: ", paste(available_sheets, collapse = ", "), sep=""))
}
cat("CAN: 验证通过！准备处理工作表：'", sheet_to_process, "'...\n", sep="")


# --- 第5步：读取指定的Excel数据 ---
cat("CAN: 正在读取文件 '", input_file, "' 的工作表 '", sheet_to_process, "'...\n", sep="")
tryCatch({
  original_data <- readxl::read_excel(input_file, sheet = sheet_to_process)
  cat("CAN: 数据读取成功！\n")
}, error = function(e) {
  stop(paste("错误：无法从工作表 '", sheet_to_process, "' 读取数据。\n原始错误信息: ", e$message, sep = ""))
})

# --- 第6步：计算统计数据 ---
cat("CAN: 正在计算统计数据 (结果将保留两位小数)...\n")
summary_stats <- original_data %>%
  group_by(method) %>%
  summarize(
    MH_mean = round(mean(`最大飞行高度`, na.rm = TRUE), 2),
    QL_mean = round(mean(`QL_value`, na.rm = TRUE), 2),
    MH_median = round(median(`最大飞行高度`, na.rm = TRUE), 2),
    QL_median = round(median(`QL_value`, na.rm = TRUE), 2),
    MH_max = round(max(`最大飞行高度`, na.rm = TRUE), 2),
    QL_max = round(max(`QL_value`, na.rm = TRUE), 2),
    MH_min = round(min(`最大飞行高度`, na.rm = TRUE), 2),
    QL_min = round(min(`QL_value`, na.rm = TRUE), 2),
    MH_Q1 = round(quantile(`最大飞行高度`, probs = 0.25, na.rm = TRUE), 2),
    QL_Q1 = round(quantile(`QL_value`, probs = 0.25, na.rm = TRUE), 2),
    MH_Q3 = round(quantile(`最大飞行高度`, probs = 0.75, na.rm = TRUE), 2),
    QL_Q3 = round(quantile(`QL_value`, probs = 0.75, na.rm = TRUE), 2),
    .groups = 'drop' 
  )

# --- 第7步：准备最终输出的表格 ---
method_order <- c("PM", "LHS", "CVT", "D_opt", "G_opt")
summary_stats <- summary_stats %>%
  mutate(method = factor(method, levels = method_order)) %>%
  arrange(method)

final_table <- summary_stats %>%
  select(
    `方法` = method,
    MH_均值 = MH_mean, QL_均值 = QL_mean,
    MH_中位数 = MH_median, QL_中位数 = QL_median,
    MH_最大值 = MH_max, QL_最大值 = QL_max,
    MH_最小值 = MH_min, QL_最小值 = QL_min,
    MH_Q1 = MH_Q1, QL_Q1 = QL_Q1,
    MH_Q3 = MH_Q3, QL_Q3 = QL_Q3
  )

cat("CAN: 计算完成，正在准备生成Excel文件...\n")

# --- 第8步：创建或更新Excel文件 ---
# この部分はV5.0のスマートなファイル処理ロジックを保持します
if (file.exists(output_file)) {
  cat("CAN: 检测到已有的 '", output_file, "' 文件。将向其中添加/更新工作表。\n", sep="")
  wb <- loadWorkbook(output_file)
} else {
  cat("CAN: 未检测到 '", output_file, "'。将为您创建一个新文件。\n", sep="")
  wb <- createWorkbook()
}

# 检查同名工作表是否已存在，如果存在则先移除
if (sheet_to_process %in% names(wb)) {
  cat("CAN: 工作表 '", sheet_to_process, "' 已存在，将进行更新。\n", sep="")
  removeWorksheet(wb, sheet_to_process)
}

# 添加一个新的工作表，名称与您在第3步中定义的名称相同
addWorksheet(wb, sheet_to_process)

# 写入第一级表头
header_L1 <- c("方法", "均值", "中位数", "最大值", "最小值", "Q1", "Q3")
writeData(wb, sheet_to_process, x = header_L1[1], startCol = 1, startRow = 1)
for (i in seq_along(header_L1[-1])) {
  col_start <- (i - 1) * 2 + 2
  writeData(wb, sheet_to_process, x = header_L1[i+1], startCol = col_start, startRow = 1)
  mergeCells(wb, sheet_to_process, cols = col_start:(col_start + 1), rows = 1)
}

# 写入第二级表头
header_L2 <- rep(c("MH", "QL"), 6)
writeData(wb, sheet_to_process, x = t(header_L2), startCol = 2, startRow = 2, colNames = FALSE)

# 写入您的数据，从第3行开始
writeData(wb, sheet_to_process, x = final_table, startRow = 3, colNames = FALSE)

# 设置列宽
setColWidths(wb, sheet_to_process, cols = 1:13, widths = "auto")
setColWidths(wb, sheet_to_process, cols = 1, widths = 15)

# --- 第9步：保存工作簿 ---
saveWorkbook(wb, output_file, overwrite = TRUE)

# --- 第10步：完成！ ---
cat("===================================================\n")
cat("CAN: 太棒了！一切搞定。\n")
cat("CAN: 您的分析结果已成功保存到文件 '", output_file, "' 的 '", sheet_to_process, "' 工作表中。\n", sep = "")
cat("CAN: 要处理其他工作表, 请返回脚本第3步修改 'sheet_to_process' 变量后重新运行。\n")
cat("CAN: 我热爱编码！\n")
