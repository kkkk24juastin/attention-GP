# =============================================================================
# 1. 环境与库设置
# =============================================================================
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['LC_ALL'] = 'zh_CN.UTF-8'

import numpy as np
import pandas as pd
import torch
from pyDOE2 import lhs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
import logging

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

rpy2_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='rpy2')

numpy2ri.activate()


# =============================================================================
# 2. 核心函数（TGP预测、数据加载、候选点生成）
# =============================================================================

def tgp_predict(X_train, u_train, X_candidates):
    """使用R的tgp包进行高斯过程预测"""
    r = robjects.r
    r.assign("X_train", X_train)
    r.assign("u_train", u_train)
    r.assign("X_candidates", X_candidates)
    try:
        r('suppressMessages(library(tgp))')
        r('model <- tryCatch(btgp(X_train, u_train, XX = X_candidates, verb=0), error=function(e) NULL)')
        if r('is.null(model)')[0]:
            print("警告: TGP模型构建失败，返回零均值和单位方差。")
            return np.zeros(X_candidates.shape[0]), np.ones(X_candidates.shape[0])
        r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')
        pred_mean = np.array(r("pred$mean"))
        pred_var = np.array(r("pred$var"))
        pred_var[np.isnan(pred_var) | (pred_var < 0)] = 1e-6
        pred_mean[np.isnan(pred_mean)] = np.mean(u_train) if u_train.size > 0 else 0
        return pred_mean, pred_var
    except Exception as e:
        print(f"R脚本执行出错: {e}")
        return np.zeros(X_candidates.shape[0]), np.ones(X_candidates.shape[0])


def load_data(file_path, num_rows):
    """从Excel文件加载数据"""
    data = pd.read_excel(file_path, nrows=num_rows)
    X = data.iloc[:, 2:7].values
    u = data.iloc[:, 7].values
    return X, u


def generate_candidates(n_samples, lower_bounds, upper_bounds):
    """使用拉丁超立方采样生成候选点"""
    dim = len(lower_bounds)
    X = lhs(dim, samples=n_samples)
    X_scaled = X * (np.array(upper_bounds) - np.array(lower_bounds)) + np.array(lower_bounds)
    return X_scaled


def ga_optimize(X_train, u_train, target_value):
    r = robjects.r
    r.assign("X_train", X_train)  # 传递训练输入
    r.assign("u_train", u_train)  # 传递训练响应
    r.assign("target_value", target_value)  # 传递目标值
    r('library(GA)')  # 加载遗传算法包
    r('library(tgp)')  # 加载tgp包
    r('''
    fitnessfun <- function(x) {
        x <- t(x)
        btgpforopt <- btgp(X=X_train, Z=u_train, XX=x, verb=0)
        mu <- btgpforopt$ZZ.mean
        var <- btgpforopt$ZZ.s2
        value <- (mu - target_value)^2 + var
        return(value)
    }
    ga_model <- ga(
        type = "real-valued",
        fitness = function(x) -fitnessfun(x),
        lower = c(0, 2, 0, 30, 1),
        upper = c(20, 5, 1, 50, 2),
        popSize = 50,
        maxiter = 200,
        run = 10,
        parallel = FALSE
    )
    best_solution <- ga_model@solution
    ''')  # 定义适应度函数并运行遗传算法
    best_solution = np.array(r('best_solution'))  # 获取最优解
    return best_solution


# =============================================================================
# 3. 主动学习相关函数 (PM, G-opt, D-opt, K-means)
# =============================================================================

def compute_space_filling_r_style(X_candidates, X_train):
    if X_train.shape[0] == 0: return np.ones(X_candidates.shape[0])
    space_filling = np.min(cdist(X_candidates, X_train), axis=1)
    max_dist = np.max(space_filling)
    return space_filling / max_dist if max_dist > 0 else np.ones(X_candidates.shape[0])


def space_filling_to_softmax(space_filling):
    space_filling_tensor = torch.tensor(np.log(space_filling), dtype=torch.float32)
    return torch.softmax(space_filling_tensor, dim=0).numpy()


def comsoftmax(X_train, u_train, X_candidates, target_value):
    mu, sigma = tgp_predict(X_train, u_train, X_candidates)
    response_penalty = (mu - target_value) ** 2 + sigma
    max_penalty = np.max(response_penalty)
    response_penalty_norm = response_penalty / max_penalty if max_penalty > 0 else np.ones_like(response_penalty)
    response_penalty_neg = -np.log(response_penalty_norm)
    softmaxpointres = torch.softmax(torch.tensor(response_penalty_neg, dtype=torch.float32), dim=0)
    return softmaxpointres.numpy(), mu


def select_d_optimal_points(X_initial, X_candidates, n_additional):
    X_initial_design = np.hstack([np.ones((X_initial.shape[0], 1)), X_initial])
    X_candidates_design = np.hstack([np.ones((X_candidates.shape[0], 1)), X_candidates])
    current_X = X_initial_design
    selected_indices_in_candidates = []
    available_indices = list(range(X_candidates_design.shape[0]))
    for _ in range(n_additional):
        max_increase = -np.inf
        best_candidate_index, best_local_index = -1, -1
        current_logdet = np.linalg.slogdet(current_X.T @ current_X)[1]
        if np.isinf(current_logdet): break
        for i, local_idx in enumerate(available_indices):
            new_X = np.vstack([current_X, X_candidates_design[local_idx]])
            sign, new_logdet = np.linalg.slogdet(new_X.T @ new_X)
            if sign <= 0: continue
            increase = new_logdet - current_logdet
            if increase > max_increase:
                max_increase, best_candidate_index, best_local_index = increase, local_idx, i
        if best_candidate_index != -1:
            selected_indices_in_candidates.append(best_candidate_index)
            current_X = np.vstack([current_X, X_candidates_design[best_candidate_index]])
            available_indices.pop(best_local_index)
    return X_candidates[selected_indices_in_candidates]


# --- 封装的策略函数 ---
def run_pm_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations, n_preselect, target_value):
    print("--- 运行 主动学习 (PM) 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    pool_indices = np.arange(X_pool.shape[0])
    for iter in range(n_iterations):
        X_remaining = X_pool[pool_indices]
        softmax_weights, _ = comsoftmax(X_train, u_train, X_remaining, target_value)
        top_indices_local = np.argsort(-softmax_weights)[:n_preselect]
        X_preselected = X_remaining[top_indices_local]
        space_filling = compute_space_filling_r_style(X_preselected, X_train)
        space_filling_weights = space_filling_to_softmax(space_filling)
        best_preselect_idx_local = np.argmax(space_filling_weights)
        best_idx_in_remaining = top_indices_local[best_preselect_idx_local]
        global_idx = pool_indices[best_idx_in_remaining]
        X_new = X_pool[global_idx:global_idx + 1, :]
        u_new = u_pool[global_idx:global_idx + 1]
        X_train = np.vstack([X_train, X_new])
        u_train = np.concatenate([u_train, u_new])
        pool_indices = np.delete(pool_indices, best_idx_in_remaining)
    return X_train, u_train


def run_g_opt_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations):
    print("--- 运行 G-最优 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    pool_indices = np.arange(X_pool.shape[0])
    for _ in range(n_iterations):
        X_remaining = X_pool[pool_indices]
        _, var_candidates = tgp_predict(X_train, u_train, X_remaining)
        idx_max_var_local = np.argmax(var_candidates)
        global_idx = pool_indices[idx_max_var_local]
        X_new = X_pool[global_idx:global_idx + 1, :]
        u_new = u_pool[global_idx:global_idx + 1]
        X_train = np.vstack([X_train, X_new])
        u_train = np.append(u_train, u_new)
        pool_indices = np.delete(pool_indices, idx_max_var_local)
    return X_train, u_train


def run_d_opt_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations):
    print("--- 运行 D-最优 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    X_additional = select_d_optimal_points(X_train, X_pool, n_iterations)
    pool_map = {tuple(row): i for i, row in enumerate(X_pool)}
    u_additional = np.array([u_pool[pool_map[tuple(point)]] for point in X_additional])
    X_final = np.vstack([X_train, X_additional])
    u_final = np.concatenate([u_train, u_additional])
    return X_final, u_final


def run_kmeans_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations):
    print("--- 运行 K-means 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    kmeans = KMeans(n_clusters=n_iterations)
    kmeans.fit(X_pool)
    new_pts = kmeans.cluster_centers_
    u_additional = []
    for point in new_pts:
        distances_to_center = np.linalg.norm(X_pool - point, axis=1)
        u_additional.append(u_pool[np.argmin(distances_to_center)])
    X_final = np.vstack([X_train, new_pts])
    u_final = np.concatenate([u_train, np.array(u_additional)])
    return X_final, u_final


# =============================================================================
# 4. 主函数和实验流程 (已按新要求重构)
# =============================================================================
def main(methods_to_run=['G_opt', 'D_opt', 'K_means']):
    # --- 实验参数设置 ---
    n_cycles = 30
    n_initial = 60
    n_iterations = 20
    n_pool_samples = 10000
    n_preselect = 5
    target_value = 400.0
    data_file = Path("rocketdata2.xlsx")
    output_file = Path("rocket_results_pm_framework80.xlsx")
    lower_bounds = [0, 2, 0, 30, 1]
    upper_bounds = [20, 5, 1, 50, 2]

    all_results = []

    for cycle in range(n_cycles):
        print(f"\n{'=' * 25} 循环 {cycle + 1}/{n_cycles} {'=' * 25}")
        X_train_initial, u_train_initial = load_data(data_file, n_initial)

        print("为当前循环生成共享候选池...")
        X_pool = generate_candidates(n_pool_samples, lower_bounds, upper_bounds)
        u_pool, _ = tgp_predict(X_train_initial, u_train_initial, X_pool)

        # --- Step 1: 运行PM方法以创建统一的评估框架 ---
        print("\n--- Step 1: 运行PM方法以创建评估框架 ---")
        X_train_pm, u_train_pm = run_pm_method(
            X_train_initial, u_train_initial, X_pool, u_pool,
            n_iterations, n_preselect, target_value
        )
        print(f"PM框架创建完毕。框架模型训练集大小: {X_train_pm.shape[0]}")
        print("--- 框架创建完成 ---")

        # --- Step 2: 运行并评估指定的方法 ---
        print("\n--- Step 2: 运行并评估其他方法 ---")
        for method in methods_to_run:
            X_train_final, u_train_final = None, None

            if method == 'G_opt':
                X_train_final, u_train_final = run_g_opt_method(X_train_initial, u_train_initial, X_pool, u_pool,
                                                                n_iterations)
            elif method == 'D_opt':
                X_train_final, u_train_final = run_d_opt_method(X_train_initial, u_train_initial, X_pool, u_pool,
                                                                n_iterations)
            elif method == 'K_means':
                X_train_final, u_train_final = run_kmeans_method(X_train_initial, u_train_initial, X_pool, u_pool,
                                                                 n_iterations)

            if X_train_final is not None:
                print(f"\n方法 '{method}' 完成采样，最终训练集大小: {X_train_final.shape[0]}")

                # Step 2a: 使用方法各自的训练集进行遗传算法优化，找到最优解
                print(f"为 '{method}' 的最终模型运行遗传算法优化...")
                result_ga = ga_optimize(X_train_final, u_train_final, target_value)

                # Step 2b: 使用PM生成的模型作为统一评估框架，来计算最优解的质量
                print(f"使用PM框架模型评估 '{method}' 找到的最优点...")
                mu_ga, var_ga = tgp_predict(X_train_pm, u_train_pm, result_ga)

                QL_value = (mu_ga - target_value) ** 2 + var_ga
                print(f"'{method}' 的最终QL值为: {QL_value[0]:.4f}")

                result_dict = {"cycle": cycle + 1, "method": method}
                for i in range(result_ga.shape[1]):
                    result_dict[f"result_ga_x{i + 1}"] = result_ga[0, i]
                result_dict["QL_value"] = float(QL_value[0])
                all_results.append(result_dict)

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_file, index=False)
    print(f"\n实验完成！所有结果已保存到: {output_file}")


if __name__ == "__main__":
    # 按照新流程，运行G_opt, D_opt, K_means三种方法，并使用PM方法构建的模型作为评估框架
    main(methods_to_run=['G_opt', 'D_opt', 'K_means'])
