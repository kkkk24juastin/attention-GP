import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
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

# -------------------------------
# 2. 利用 rpy2 调用 R 包 tgp 实现 btgp 模型预测
def tgp_predict(X_train, u_train, X_candidates):
    r = robjects.r
    r.assign("X_train", X_train)
    r.assign("u_train", u_train)
    r.assign("X_candidates", X_candidates)
    r('library(tgp)')
    r('model <- btgp(X_train, u_train, XX = X_candidates)')
    r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')
    pred_mean = np.array(r("pred$mean"))
    pred_var = np.array(r("pred$var"))
    return pred_mean, pred_var

# -------------------------------
# 3. 参考R代码实现的空间填充度计算
def compute_space_filling_r_style(X_candidates, X_train):
    n_candidates = X_candidates.shape[0]
    space_filling = np.zeros(n_candidates)
    for j in range(n_candidates):
        candidate_matrix = np.tile(X_candidates[j], (X_train.shape[0], 1))
        space = candidate_matrix - X_train
        distances = np.sqrt(np.sum(space ** 2, axis=1))
        space_filling[j] = np.min(distances)
    space_filling = space_filling / max(space_filling)
    return space_filling

def space_filling_to_softmax(space_filling):
    space_filling = np.log(space_filling)
    space_filling_tensor = torch.tensor(space_filling, dtype=torch.float32)
    softmax_weights = torch.softmax(space_filling_tensor, dim=0)
    return softmax_weights.numpy()

# -------------------------------
# 4. 构造候选点特征和响应惩罚
def compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value, epsilon=0):
    mu, sigma = tgp_predict(X_train, u_train, X_candidates)
    response_penalty = (mu - target_value) ** 2 + sigma
    response_penalty = response_penalty / max(response_penalty)
    return mu, sigma, response_penalty

def comsoftmax(X_train, u_train, X_candidates, target_value, epsilon=0):
    mu, sigma, response_penalty = compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value, epsilon)
    response_penalty_neg = -np.log(response_penalty)
    log_norm_penalty_tensor = torch.tensor(response_penalty_neg, dtype=torch.float32)
    softmaxpointres = torch.softmax(log_norm_penalty_tensor, dim=0)
    return softmaxpointres.numpy(), mu, sigma, response_penalty

if __name__ == "__main__":
    n = 20  # 循环执行的次数
    n_iterations = 30  # 主动学习迭代次数
    n_preselect = 50  # 预选点数量
    epsilon_val = 0  # 用于数值稳定的小常数
    target_value = 30

    # 创建列表存储 RMSE 结果
    rmse_act_list = []
    rmse_rad_list = []

    for run in range(n):
        print(f"\n===================== 执行第 {run + 1}/{n} 次 =====================")

        # 加载数据
        X_TEST, u_test = load_data("data.xlsx")
        X_train, u_train = load_data("acttrain.xlsx")
        X_pool, u_pool = load_data("actpool.xlsx")
        X_rad, u_rad = load_data("lhstrain.xlsx")

        n_initial = X_train.shape[0]
        n_pool = X_pool.shape[0]

        remaining_indices = list(range(n_pool))
        X_remaining = X_pool.copy()

        for iter in range(n_iterations):
            softmax_weights, mu, sigma, response_penalty = comsoftmax(X_train, u_train, X_remaining, target_value, epsilon_val)
            top_indices = np.argsort(-softmax_weights)[:n_preselect]
            X_preselected = X_remaining[top_indices]
            space_filling = compute_space_filling_r_style(X_preselected, X_train)
            space_filling_weights = space_filling_to_softmax(space_filling)
            best_preselect_idx = np.argmax(space_filling_weights)
            best_local_idx = top_indices[best_preselect_idx]
            best_global_index = remaining_indices[best_local_idx]
            X_new = X_pool[best_global_index].reshape(1, -1)
            u_new = u_pool[best_global_index].reshape(1, )
            X_train = np.vstack([X_train, X_new])
            u_train = np.concatenate([u_train, u_new])
            remaining_indices.remove(best_global_index)
            X_remaining = X_pool[remaining_indices]

        muact, s2act = tgp_predict(X_train, u_train, X_TEST)
        murad, s2rad = tgp_predict(X_rad, u_rad, X_TEST)

        mask = (u_test >= 25) & (u_test <= 35)
        u_true = u_test[mask]
        mu_act = muact[mask]
        mu_rad = murad[mask]

        rmse_act = np.sqrt(np.mean((u_true - mu_act) ** 2))
        rmse_rad = np.sqrt(np.mean((u_true - mu_rad) ** 2))

        rmse_act_list.append(rmse_act)
        rmse_rad_list.append(rmse_rad)

        print(f"第 {run + 1} 次执行：主动学习 RMSE = {rmse_act:.4f}, LHS RMSE = {rmse_rad:.4f}")

    # 将 RMSE 结果保存到 XLSX 文件
    rmse_df = pd.DataFrame({
        '主动学习 RMSE': rmse_act_list,
        'LHS RMSE': rmse_rad_list
    })
    rmse_df.to_excel('rmse_results.xlsx', index=False)
    print("\nRMSE 结果已保存到 'rmse_results.xlsx' 文件中。")
