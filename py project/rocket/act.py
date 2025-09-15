# 环境设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免库冲突（如TensorFlow或PyTorch）
os.environ['LC_ALL'] = 'zh_CN.UTF-8'         # 设置中文环境支持

# 数据处理和科学计算
import numpy as np                           # 数组操作和数值计算
import pandas as pd                          # 数据加载和处理
from pyDOE2 import lhs                       # 拉丁超立方采样生成候选点

# 深度学习
import torch                                 # 用于softmax计算

# 文件路径管理
from pathlib import Path                     # 跨平台路径处理

# 绘图设置
import matplotlib.pyplot as plt              # 数据可视化
import matplotlib
matplotlib.use("TkAgg")                      # 设置绘图后端
plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

# R接口模块
import rpy2.robjects as robjects             # Python与R交互
from rpy2.robjects import numpy2ri           # NumPy与R数据转换
numpy2ri.activate()                          # 激活NumPy到R的自动转换

# 函数：tgp_predict
# 描述：使用R的tgp包进行高斯过程预测
# 参数：
#   X_train: 训练集输入，形状为(n_samples, n_features)的numpy数组
#   u_train: 训练集响应，形状为(n_samples,)的numpy数组
#   X_candidates: 候选点，形状为(n_candidates, n_features)的numpy数组
# 返回：
#   pred_mean: 预测均值，形状为(n_candidates,)的numpy数组
#   pred_var: 预测方差，形状为(n_candidates,)的numpy数组
def tgp_predict(X_train, u_train, X_candidates):
    r = robjects.r
    r.assign("X_train", X_train)         # 将训练输入传递给R
    r.assign("u_train", u_train)         # 将训练响应传递给R
    r.assign("X_candidates", X_candidates)  # 将候选点传递给R
    r('library(tgp)')                    # 加载R的tgp包
    r('model <- btgp(X_train, u_train, XX = X_candidates)')  # 训练高斯过程模型
    r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')  # 提取预测结果
    pred_mean = np.array(r("pred$mean"))  # 获取预测均值
    pred_var = np.array(r("pred$var"))    # 获取预测方差
    return pred_mean, pred_var

# 函数：load_data
# 描述：从Excel文件加载数据
# 参数：
#   file_path: Excel文件路径（Path对象）
#   num_rows: 要读取的行数
# 返回：
#   X: 输入特征，形状为(num_rows, 5)的numpy数组（第3-7列）
#   u: 响应值，形状为(num_rows,)的numpy数组（第8列）
def load_data(file_path, num_rows):
    data = pd.read_excel(file_path, nrows=num_rows)  # 读取指定行数的Excel数据
    X = data.iloc[:, 2:7].values  # 提取第3到7列作为特征（索引从0开始）
    u = data.iloc[:, 7].values    # 提取第8列作为响应值
    return X, u

# 函数：generate_candidates
# 描述：使用拉丁超立方采样生成候选点
# 参数：
#   n_samples: 生成的样本数量
#   lower_bounds: 每个维度的下界（列表）
#   upper_bounds: 每个维度的上界（列表）
# 返回：
#   X: 候选点，形状为(n_samples, n_features)的numpy数组
def generate_candidates(n_samples, lower_bounds, upper_bounds):
    dim = len(lower_bounds)  # 获取维度数
    X = lhs(dim, samples=n_samples)  # 生成[0,1]区间内的样本
    X = X * (np.array(upper_bounds) - np.array(lower_bounds)) + np.array(lower_bounds)  # 缩放到指定范围
    return X

# 函数：compute_space_filling_r_style
# 描述：计算候选点相对于训练点的空间填充度（最小距离）
# 参数：
#   X_candidates: 候选点，形状为(n_candidates, n_features)的numpy数组
#   X_train: 训练点，形状为(n_train, n_features)的numpy数组
# 返回：
#   space_filling: 空间填充度，形状为(n_candidates,)的numpy数组
def compute_space_filling_r_style(X_candidates, X_train):
    n_candidates = X_candidates.shape[0]
    space_filling = np.zeros(n_candidates)
    for j in range(n_candidates):
        candidate_matrix = np.tile(X_candidates[j], (X_train.shape[0], 1))  # 复制候选点
        space = candidate_matrix - X_train  # 计算与训练点的差
        distances = np.sqrt(np.sum(space ** 2, axis=1))  # 计算欧几里得距离
        space_filling[j] = np.min(distances)  # 取最小距离
    space_filling = space_filling / np.max(space_filling)  # 归一化
    return space_filling

# 函数：spaceiony_to_softmax
# 描述：将空间填充度转换为softmax权重
# 参数：
#   space_filling: 空间填充度，形状为(n_candidates,)的numpy数组
# 返回：
#   softmax_weights: softmax权重，形状为(n_candidates,)的numpy数组
def space_filling_to_softmax(space_filling):
    space_filling = np.log(space_filling)  # 对空间填充度取对数（越大越好）
    space_filling_tensor = torch.tensor(space_filling, dtype=torch.float32)  # 转换为PyTorch张量
    softmax_weights = torch.softmax(space_filling_tensor, dim=0)  # 计算softmax权重
    return softmax_weights.numpy()  # 转换回numpy数组

# 函数：compute_candidate_features_tgp
# 描述：计算候选点的预测特征（均值、方差、响应惩罚）
# 参数：
#   X_train: 训练集输入，形状为(n_train, n_features)的numpy数组
#   u_train: 训练集响应，形状为(n_train,)的numpy数组
#   X_candidates: 候选点，形状为(n_candidates, n_features)的numpy数组
#   target_value: 目标响应值
#   epsilon: 数值稳定常数（默认0，未使用）
# 返回：
#   mu: 预测均值，形状为(n_candidates,)的numpy数组
#   sigma: 预测方差，形状为(n_candidates,)的numpy数组
#   response_penalty: 响应惩罚，形状为(n_candidates,)的numpy数组
def compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value, epsilon=0):
    mu, sigma = tgp_predict(X_train, u_train, X_candidates)  # 获取预测均值和方差
    response_penalty = (mu - target_value) ** 2 + sigma  # 计算响应惩罚
    response_penalty = response_penalty / np.max(response_penalty)  # 归一化
    return mu, sigma, response_penalty

# 函数：comsoftmax
# 描述：基于响应惩罚计算softmax权重
# 参数：
#   X_train: 训练集输入，形状为(n_train, n_features)的numpy数组
#   u_train: 训练集响应，形状为(n_train,)的numpy数组
#   X_candidates: 候选点，形状为(n_candidates, n_features)的numpy数组
#   target_value: 目标响应值
#   epsilon: 数值稳定常数（默认0，未使用）
# 返回：
#   softmaxpointres: softmax权重，形状为(n_candidates,)的numpy数组
#   mu: 预测均值，形状为(n_candidates,)的numpy数组
#   sigma: 预测方差，形状为(n_candidates,)的numpy数组
#   response_penalty: 响应惩罚，形状为(n_candidates,)的numpy数组
def comsoftmax(X_train, u_train, X_candidates, target_value, epsilon=0):
    mu, sigma, response_penalty = compute_candidate_features_tgp(
        X_train, u_train, X_candidates, target_value, epsilon
    )
    response_penalty_neg = -np.log(response_penalty)  # 取负对数（惩罚越小权重越高）
    log_norm_penalty_tensor = torch.tensor(response_penalty_neg, dtype=torch.float32)  # 转换为张量
    softmaxpointres = torch.softmax(log_norm_penalty_tensor, dim=0)  # 计算softmax权重
    return softmaxpointres.numpy(), mu, sigma, response_penalty

if __name__ == "__main__":
    # 定义参数
    n_iterations = 20  # 主动学习迭代次数
    n_preselect = 5  # 每轮预选的候选点数量
    epsilon_val = 0  # 数值稳定常数（未使用）
    target_value = 400  # 目标响应值
    file = Path(r"C:/Users/kkkk24/Desktop/backupfiles/py project/rocket/rocketdata.xlsx")  # 数据文件路径

    # 加载初始训练数据
    X_train, u_train = load_data(file, 30)  # 读取前30行作为初始训练集
    n_initial = X_train.shape[0]  # 初始训练点数量

    # 定义循环次数
    n_cycles = 1  # 重复执行20次
    results = []  # 存储每次循环的结果

    # 主循环
    for cycle in range(n_cycles):
        print(f"\n===================== 循环 {cycle + 1}/{n_cycles} =====================")

        # 加载初始训练数据（每次循环重新加载，避免累积）
        X_train, u_train = load_data(file, 30)
        n_initial = X_train.shape[0]

        # 生成候选点池
        n_samples = 10000
        lower_bounds = [0, 2, 0, 30, 1]
        upper_bounds = [20, 5, 1, 50, 2]
        X_pool = generate_candidates(n_samples, lower_bounds, upper_bounds)
        u_pool, _ = tgp_predict(X_train, u_train, X_pool)
        n_pool = X_pool.shape[0]

        # 输出初始训练点信息
        print("\n===================== 初始设计点及响应 =====================")
        for i in range(n_initial):
            print(f"初始点 {i + 1}: X=({X_train[i, 0]:.4f}, {X_train[i, 1]:.4f}, {X_train[i, 2]:.4f}), "
                  f"响应={u_train[i]:.4f}")

        # 初始化候选池管理
        remaining_indices = list(range(n_pool))
        X_remaining = X_pool.copy()
        added_points_info = []

        # 主动学习循环
        print(f"\n开始 {n_iterations} 轮主动学习迭代...")
        for iter in range(n_iterations):
            print(f"\n================ 迭代 {iter + 1}/{n_iterations} ================")
            softmax_weights, mu, sigma, response_penalty = comsoftmax(
                X_train, u_train, X_remaining, target_value, epsilon_val
            )
            top_indices = np.argsort(-softmax_weights)[:n_preselect]
            print("\n========== 预选点的权重值和响应值 ==========")
            print("序号\t权重值\t预测响应值")
            for i, idx in enumerate(top_indices):
                print(f"{i + 1}\t{softmax_weights[idx]:.6f}\t{mu[idx]:.6f}")
            X_preselected = X_remaining[top_indices]
            space_filling = compute_space_filling_r_style(X_preselected, X_train)
            space_filling_weights = space_filling_to_softmax(space_filling)
            print("\n========== 预选点的空间填充度和权重 ==========")
            print("序号\t空间填充度\t空间填充权重")
            for i in range(len(space_filling)):
                print(f"{i + 1}\t{space_filling[i]:.6f}\t{space_filling_weights[i]:.6f}")
            best_preselect_idx = np.argmax(space_filling_weights)
            best_local_idx = top_indices[best_preselect_idx]
            best_global_index = remaining_indices[best_local_idx]
            X_new = X_pool[best_global_index].reshape(1, -1)
            u_new = u_pool[best_global_index].reshape(1, )
            print(f"--> 增加点: X=({X_new[0, 0]:.4f}, {X_new[0, 1]:.4f}, {X_new[0, 2]:.4f}), "
                  f"响应={u_new[0]:.4f}, 空间填充度: {space_filling[best_preselect_idx]:.6f}")
            added_points_info.append({
                "iteration": iter + 1,
                "coordinates": tuple(X_new[0].tolist()),  # 转换为元组以便存储
                "response": float(u_new[0]),  # 转换为标量
                "space_filling": space_filling[best_preselect_idx]
            })
            X_train = np.vstack([X_train, X_new])
            u_train = np.concatenate([u_train, u_new])
            remaining_indices.remove(best_global_index)
            X_remaining = X_pool[remaining_indices]

        # 记录本次循环的主动学习结果
        results.append({
            "cycle": cycle + 1,
            "added_points": added_points_info
        })

        # 输出主动学习结果
        print("\n===================== 主动学习结束 =====================")
        print(f"最终训练集包含 {X_train.shape[0]} 个点。")

    # 保存结果到 Excel
    # 由于 added_points 是列表，Excel 无法直接存储列表，需要展平或以其他方式处理
    # 这里我们将每个循环的 added_points 信息展平为多行
    rows = []
    for result in results:
        cycle = result["cycle"]
        for point in result["added_points"]:
            rows.append({
                "cycle": cycle,
                "iteration": point["iteration"],
                "coordinates": str(point["coordinates"]),  # 转换为字符串
                "response": point["response"],
                "space_filling": point["space_filling"]
            })

    df_results = pd.DataFrame(rows)
    output_file = Path(r"C:/Users/kkkk24/Desktop/backupfiles/py project/rocket/active_learning_results.xlsx")
    df_results.to_excel(output_file, index=False)
    print(f"\n主动学习结果已保存到 {output_file}")
