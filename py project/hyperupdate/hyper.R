# CAN: 完整脚本 - 运行btgp并导出所有超参数轨迹

# --------------------------------------------------------------------------
# 步骤 1: 加载所有必需的库
# --------------------------------------------------------------------------
# (保留您原有的库)
library(tgp)
library(lhs)
# (添加用于写入Excel的库)
if (!require("openxlsx")) {
  install.packages("openxlsx")
  library(openxlsx)
}

cat("所有必要的库已加载。\n")


# --------------------------------------------------------------------------
# 步骤 2: 您提供的函数和数据生成部分 (保持原样)
# --------------------------------------------------------------------------
slhd_xiu <- function(m, n, L, U) {
  xtrain <- matrix(0, nrow = m, ncol = n)
  for (i in 1:n) {
    X_train <- lhs::randomLHS(m, 1) # 使用随机拉丁超立方采样
    XF1 <- X_train[, 1]
    xtrain[, i] <- (U[i] - L[i]) * XF1 + L[i]
  }
  return(xtrain)
}

non_test_function <- function(x, y) {
  step_function <- ifelse(x > 0, 1, 0)
  result <- ifelse(
    step_function == 1,
    2 * sin(x) + y^2,
    x^2 + y^2
  )
  return(result)
}

m <- 40 # 样本数量
n <- 2 # 维度数量
L <- c(-2, -2) # 下限向量
U <- c(2, 2) # 上限向量

TRUETEST <- function(X) {
  x <- X[, 1]
  y <- X[, 2]
  Z <- non_test_function(x, y)
  return(Z)
}

# 生成训练和测试数据 (与您提供的一致)
X <- slhd_xiu(m, n, L, U)
Z <- TRUETEST(X)
Xt <- slhd_xiu(1, n, L, U)

# 为了让列名更有意义，在拟合前给它们命名
colnames(X) <- c("x", "y")
colnames(Xt) <- c("x", "y")


cat("自定义函数和数据已准备就绪。\n")

# --------------------------------------------------------------------------
# 步骤 3: 运行 btgp 拟合 - 这是我们进行修改的核心部分
# --------------------------------------------------------------------------
cat("正在运行btgp模型，默认迭代次数为7000，这可能需要一点时间...\n")

# 在您的调用中添加 trace = TRUE
btgppre <- btgp(
  X = X, 
  Z = Z, 
  XX = Xt,
  BTE = c(2000, 22000, 2),
  trace = TRUE  # <--- 这是激活MCMC轨迹记录的关键修改！
)

cat("模型拟合完成！MCMC轨迹已成功捕获。\n")


# --------------------------------------------------------------------------
# 步骤 4: 新增代码块 - 提取轨迹并保存到 XLSX 文件
# --------------------------------------------------------------------------
# 定义输出的Excel文件名
output_filename <- "my_btgp_trace1.xlsx"

# 创建一个新的Excel工作簿
wb <- createWorkbook()

# 从拟合结果中提取轨迹列表
mcmc_trace <- btgppre$trace

cat("正在将轨迹数据整理并写入Excel文件...\n")

# 将每个超参数的轨迹写入一个单独的工作表
for (param_name in names(mcmc_trace)) {
  
  # 将轨迹数据转换为数据框格式
  param_data <- as.data.frame(mcmc_trace[[param_name]])
  
  # 为相关性参数 'd' 的列提供明确的名称
  if (param_name == "d") {
    # 使用输入数据的列名来命名，更具可读性
    colnames(param_data) <- paste0("d_", colnames(X)) 
  }
  
  # 为均值参数 'beta' 的列提供名称
  if (param_name == "beta") {
    colnames(param_data) <- paste0("beta_", 0:(ncol(param_data)-1)) # beta_0, beta_1, etc.
  }
  
  # 在工作簿中添加一个新工作表
  addWorksheet(wb, param_name)
  
  # 将数据写入该工作表
  writeData(wb, sheet = param_name, x = param_data)
}

# 保存工作簿到文件
saveWorkbook(wb, file = output_filename, overwrite = TRUE)

# 打印最终成功消息，并告知文件位置
cat(paste0("\n--- 任务完成 ---\n"))
cat(paste0("所有超参数的MCMC迭代历史已成功保存到文件：\n", file.path(getwd(), output_filename), "\n"))
cat(paste0("该文件包含 ", length(names(mcmc_trace)), " 个工作表，每个工作表包含 ", nrow(mcmc_trace$s2), " 次MCMC迭代的样本。\n"))
