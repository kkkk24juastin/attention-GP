import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd


from scipy.spatial.distance import cdist
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 导入 rpy2 相关模块
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

numpy2ri.activate()

# -------------------------------
# 1. 读取初始训练集和候选点
def load_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 0:3].values  # 第2-4列为输入特征
    u = data.iloc[:, 3].values    # 第5列为响应值
    return X, u
X_TEST,u_test=load_data("data.xlsx")
# 加载初始训练集
X_train, u_train = load_data("acttrain.xlsx")
n_initial = X_train.shape[0]  # 初始训练点数量

# 加载候选点池
X_pool, u_pool = load_data("actpool.xlsx")
n_pool = X_pool.shape[0]  # 候选池大小
# 加载lhs训练集
X_rad, u_rad = load_data("lhstrain.xlsx")


# -------------------------------
# 2. 利用 rpy2 调用 R 包 tgp 实现 btgp 模型预测
def tgp_predict(X_train, u_train, X_candidates):
    r = robjects.r
    # 将数据赋值到 R 环境中
    r.assign("X_train", X_train)
    r.assign("u_train", u_train)
    r.assign("X_candidates", X_candidates)

    # 执行 R 代码：加载 tgp 包，调用 btgp 函数，并将预测结果保存在变量 pred
    r('library(tgp)')
    r('model <- btgp(X_train, u_train, XX = X_candidates)')
    r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')

    # 从 R 中提取预测均值与方差（注意：此处 sigma 为方差）
    pred_mean = np.array(r("pred$mean"))
    pred_var = np.array(r("pred$var"))

    return pred_mean, pred_var

# -------------------------------
# 3. 参考R代码实现的空间填充度计算
def compute_space_filling_r_style(X_candidates, X_train):
    n_candidates = X_candidates.shape[0]
    space_filling = np.zeros(n_candidates)

    for j in range(n_candidates):
        # 复制当前候选点成与训练点数量一样的矩阵
        candidate_matrix = np.tile(X_candidates[j], (X_train.shape[0], 1))
        # 计算与所有训练点的差
        space = candidate_matrix - X_train
        # 计算欧几里得距离
        distances = np.sqrt(np.sum(space ** 2, axis=1))
        # 取最小距离作为空间填充度
        space_filling[j] = np.min(distances)
    space_filling = space_filling / max(space_filling)
    return space_filling

def space_filling_to_softmax(space_filling):
    # 将空间填充度转换为tensor
    # 注意：空间填充度越大越好，所以不需要取负
    space_filling = np.log(space_filling)
    space_filling_tensor = torch.tensor(space_filling, dtype=torch.float32)
    # 应用softmax获取权重
    softmax_weights = torch.softmax(space_filling_tensor, dim=0)
    return softmax_weights.numpy()

# -------------------------------
# 4. 构造候选点特征和响应惩罚
def compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value, epsilon=0):
    # 利用 tgp_predict 获得预测结果
    mu, sigma = tgp_predict(X_train, u_train, X_candidates)

    # 响应惩罚
    response_penalty = (mu - target_value) ** 2 + sigma
    response_penalty = response_penalty / max(response_penalty)
    return mu, sigma, response_penalty

def comsoftmax(X_train, u_train, X_candidates, target_value, epsilon=0):
    mu, sigma, response_penalty = compute_candidate_features_tgp(X_train, u_train, X_candidates,
                                                                 target_value=target_value,
                                                                 epsilon=epsilon)
    # 将响应惩罚转换为权重 (越小越好，所以取负)
    response_penalty_neg = -np.log(response_penalty)
    # 使用softmax获取权重
    log_norm_penalty_tensor = torch.tensor(response_penalty_neg, dtype=torch.float32)
    softmaxpointres = torch.softmax(log_norm_penalty_tensor, dim=0)

    return softmaxpointres.numpy(), mu, sigma, response_penalty

if __name__ == "__main__":
    # 定义参数
    n_iterations = 30  # 主动学习迭代次数
    n_preselect = 50  # 预选点数量
    epsilon_val = 0  # 用于数值稳定的小常数
    target_value=30
    # 初始训练集和候选点池
    print(f"加载初始训练集: {n_initial} 个点")
    print(f"加载候选点池: {n_pool} 个点")

    # 打印初始设计点及其响应
    print("\n===================== 初始设计点及响应 =====================")
    for i in range(n_initial):
        print(f"初始点 {i + 1}: X=({X_train[i, 0]:.4f}, {X_train[i, 1]:.4f}, {X_train[i, 2]:.4f}), 响应={u_train[i]:.4f}")

    # 剩余候选点索引及数据
    remaining_indices = list(range(n_pool))
    X_remaining = X_pool.copy()

    # 创建列表存储所有新增点的信息
    added_points_info = []

    # 主动学习循环：每轮先选择500个预选点，再利用空间填充度选择最终点
    print(f"\n开始 {n_iterations} 轮主动学习迭代...")
    for iter in range(n_iterations):
        print(f"\n================ 迭代 {iter + 1}/{n_iterations} ================")

        # 1. 计算剩余候选点的选择权重和响应惩罚
        print(f"计算 {len(remaining_indices)} 个剩余候选点的选择权重...")
        softmax_weights, mu, sigma, response_penalty = comsoftmax(X_train, u_train, X_remaining, target_value=target_value , epsilon=epsilon_val)
        # 2. 选择响应惩罚最小的500个点 (或者softmax权重最高的500个点)
        # 这里我们使用响应惩罚，因为更直观（值越小越好）
        print(f"选择响应惩罚最小的 {n_preselect} 个点...")
        top_indices = np.argsort(-softmax_weights)[:n_preselect]  # 从小到大排序，取前n_preselect个
        print("\n========== 预选点的权重值和响应值 ==========")
        print("序号\t权重值\t预测响应值")
        # 获取预选点的权重
        selected_weights = softmax_weights[top_indices]
        selected_responses = mu[top_indices]
        for i, idx in enumerate(top_indices):  # 只输出前10个，避免信息过多
            print(f"{i + 1}\t{softmax_weights[idx]:.6f}\t{mu[idx]:.6f}")
        print(f"计算 {n_preselect} 个预选点的空间填充度...")
        X_preselected = X_remaining[top_indices]
        space_filling = compute_space_filling_r_style(X_preselected, X_train)
        space_filling_weights = space_filling_to_softmax(space_filling)
        # 输出空间填充度及其softmax权重
        print("\n========== 预选点的空间填充度和权重 ==========")
        print("序号\t空间填充度\t空间填充权重")
        for i in range(min(50, len(space_filling))):  # 只输出前10个
            print(f"{i + 1}\t{space_filling[i]:.6f}\t{space_filling_weights[i]:.6f}")
        # 5. 选择空间填充度权重最高的点
        best_preselect_idx = np.argmax(space_filling_weights)
        best_local_idx = top_indices[best_preselect_idx]
        best_global_index = remaining_indices[best_local_idx]

        # 获取最佳点坐标
        X_new = X_pool[best_global_index].reshape(1, -1)
        u_new = u_pool[best_global_index].reshape(1, )  # 从datapool中获取响应值

        # 输出增加的点及其响应和得分
        print(
            f"--> 增加点: X=({X_new[0, 0]:.4f}, {X_new[0, 1]:.4f}, {X_new[0, 2]:.4f}), 响应={u_new[0]:.4f}, 空间填充度: {space_filling[best_preselect_idx]:.6f}")

        # 6. 存储新增点信息
        added_points_info.append({
            "iteration": iter + 1,
            "coordinates": tuple(X_new[0]),
            "response": u_new[0],
            "space_filling": space_filling[best_preselect_idx]
        })

        # 7. 更新训练集
        X_train = np.vstack([X_train, X_new])
        u_train = np.concatenate([u_train, u_new])

        # 8. 从候选池中移除该点
        remaining_indices.remove(best_global_index)
        X_remaining = X_pool[remaining_indices]

    print("\n===================== 主动学习结束 =====================")
    print(f"最终训练集包含 {X_train.shape[0]} 个点。")

    print("\n===================== 所有新增点信息摘要 =====================")
    for info in added_points_info:
        print(
            f"迭代 {info['iteration']}: X=({info['coordinates'][0]:.4f}, {info['coordinates'][1]:.4f}, {info['coordinates'][2]:.4f}), 响应={info['response']:.4f}, 空间填充度={info['space_filling']:.6f}")
    muact,s2act=tgp_predict(X_train, u_train, X_TEST)
    murad,s2rad=tgp_predict(X_rad, u_rad, X_TEST)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 筛选响应在70-90之间的测试点
    mask = (u_test >= 25) & (u_test <= 35)
    X_true = X_TEST[mask]
    u_true = u_test[mask]
    mu_act = muact[mask]
    mu_rad = murad[mask]

    # 创建一个大图，包含三个子图
    fig = plt.figure(figsize=(18, 6))

    # 子图1：真实数据
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], c=u_true, cmap='viridis', vmin=70, vmax=90)
    ax1.set_title('真实数据')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('X3')

    # 子图2：主动学习方法
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], c=mu_act, cmap='viridis', vmin=70, vmax=90)
    ax2.set_title('主动学习方法')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('X3')

    # 子图3：LHS方法
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], c=mu_rad, cmap='viridis', vmin=70, vmax=90)
    ax3.set_title('LHS方法')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_zlabel('X3')

    # 添加共享的colorbar
    cbar = fig.colorbar(scatter3, ax=[ax1, ax2, ax3], shrink=0.5, aspect=5, pad=0.1)
    cbar.set_label('响应值')

    # 显示图像
    plt.show()
    import numpy as np

    # 筛选响应在70-90之间的测试点
    mask = (u_test >= 25) & (u_test <= 35)
    u_true = u_test[mask]
    mu_act = muact[mask]
    mu_rad = murad[mask]

    # 计算主动学习方法的 RMSE
    rmse_act = np.sqrt(np.mean((u_true - mu_act) ** 2))

    # 计算 LHS 方法的 RMSE
    rmse_rad = np.sqrt(np.mean((u_true - mu_rad) ** 2))

    # 打印结果
    print(f"主动学习方法在响应值 70-90 区间的 RMSE: {rmse_act:.4f}")
    print(f"LHS 方法在响应值 70-90 区间的 RMSE: {rmse_rad:.4f}")





