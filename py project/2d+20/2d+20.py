import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc
import torch
from pyDOE2 import lhs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()
import pandas as pd
from sklearn.cluster import KMeans

# 测试函数
def non_test_function(X):
    x, y = X[:, 0], X[:, 1]
    condition = x > 0
    return np.where(condition,
                    (2 + 0.5 * y) * np.sin(x) + y**2,
                    x**2 + y**2 + (8 - (x**2 + y**2)) * (1 - np.exp(-x**2)))

# 生成候选点
def generate_candidates(n_samples, lower_bounds, upper_bounds):
    lower = np.array(lower_bounds)
    upper = np.array(upper_bounds)
    dim = len(lower)
    unit_lhs = lhs(dim, samples=n_samples)
    return unit_lhs * (upper - lower) + lower

# TGP 预测函数
def tgp_predict(X_train, u_train, X_candidates):
    r = robjects.r
    r.assign("X_train", X_train)
    r.assign("u_train", u_train)
    r.assign("X_candidates", X_candidates)
    r('library(tgp)')
    r('model <- btgp(X_train, u_train, XX = X_candidates)')
    r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')
    return np.array(r("pred$mean")), np.array(r("pred$var"))

# 评估模型
def evaluate_model(X_train, u_train, Testx, Testy, target_value):
    predict_y, _ = tgp_predict(X_train, u_train, Testx)
    RMSE_all = np.sqrt(np.mean((predict_y - Testy) ** 2))
    mask = (Testy >= target_value - 0.2) & (Testy <= target_value + 0.2)
    if mask.any():
        RMSE_target = np.sqrt(np.mean((predict_y[mask] - Testy[mask]) ** 2))
    else:
        RMSE_target = np.nan
    return RMSE_all, RMSE_target

# 主动学习相关函数
def compute_space_filling_r_style(X_candidates, X_train):
    n_candidates = X_candidates.shape[0]
    space_filling = np.zeros(n_candidates)
    for j in range(n_candidates):
        distances = np.sqrt(np.sum((X_candidates[j] - X_train) ** 2, axis=1))
        space_filling[j] = np.min(distances)
    return space_filling / np.max(space_filling)

def space_filling_to_softmax(space_filling):
    space_filling = np.log(space_filling)
    space_filling_tensor = torch.tensor(space_filling, dtype=torch.float32)
    return torch.softmax(space_filling_tensor, dim=0).numpy()

def compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value):
    mu, sigma = tgp_predict(X_train, u_train, X_candidates)
    response_penalty = (mu - target_value) ** 2 + sigma
    return mu, sigma, response_penalty / np.max(response_penalty)

def comsoftmax(X_train, u_train, X_candidates, target_value):
    mu, sigma, response_penalty = compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value)
    response_penalty_neg = -np.log(response_penalty)
    log_norm_penalty_tensor = torch.tensor(response_penalty_neg, dtype=torch.float32)
    softmaxpointres = torch.softmax(log_norm_penalty_tensor, dim=0)
    return softmaxpointres.numpy(), mu, sigma, response_penalty

# D-最优设计
def d_optimality(X):
    return np.linalg.det(X.T @ X)

def select_d_optimal_points(X_initial, X_candidates, n_additional):
    X_initial_design = np.hstack([np.ones((X_initial.shape[0], 1)), X_initial])
    X_candidates_design = np.hstack([np.ones((X_candidates.shape[0], 1)), X_candidates])
    current_X = X_initial_design
    current_det = d_optimality(current_X)
    selected_indices = []
    for _ in range(n_additional):
        max_increase = -np.inf
        best_index = -1
        for i in range(X_candidates.shape[0]):
            if i not in selected_indices:
                new_X = np.vstack([current_X, X_candidates_design[i]])
                new_det = d_optimality(new_X)
                increase = new_det - current_det
                if increase > max_increase:
                    max_increase = increase
                    best_index = i
        if best_index != -1:
            selected_indices.append(best_index)
            current_X = np.vstack([current_X, X_candidates_design[best_index]])
            current_det = d_optimality(current_X)
    return X_candidates[selected_indices]

# 保存和加载进度
def save_progress(n_initial, repeat, method, methods_to_run):
    progress = {
        'n_initial': n_initial,
        'repeat': repeat,
        'method': method,
        'methods_to_run': methods_to_run
    }
    with open('progressall2d+20.json', 'w') as f:
        json.dump(progress, f)

def load_progress():
    if os.path.exists('progressall2d+20.json'):
        with open('progressall2d+20.json', 'r') as f:
            progress = json.load(f)
        return (progress['n_initial'], progress['repeat'], 
                progress['method'], progress['methods_to_run'])
    else:
        return None, None, None, None

# 追加到 Excel
def append_to_excel(df, filename):
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df
    updated_df.to_excel(filename, index=False)

# 主函数
def main(methods_to_run=['PM', 'LHS', 'G_opt', 'D_opt', 'K_means']):
    lower_bounds = [-2, -2]
    upper_bounds = [2, 2]
    target_value = 1.5
    n_pool = 2000
    n_iterations = 20
    n_preselect = 25
    n_initial_values = range(20, 81, 10)
    n_repeats = 30
    excel_filename = '2d+20五种方法结果.xlsx'
    
    # 生成固定的测试集
    Testx = generate_candidates(300, lower_bounds, upper_bounds)
    Testy = non_test_function(Testx)
    
    # 加载进度
    last_n_initial, last_repeat, last_method, last_methods_to_run = load_progress()
    if all(x is not None for x in [last_n_initial, last_repeat, last_method, last_methods_to_run]):
        print(f"从 n_initial={last_n_initial}, repeat={last_repeat+1}, "
              f"method={last_method}, methods_to_run={last_methods_to_run} 继续实验")
        methods_to_run = last_methods_to_run  # 使用保存的 methods_to_run
    else:
        print("从头开始实验")
        last_n_initial = min(n_initial_values) - 1
        last_repeat = -1
        last_method = None
    
    # 准备一个列表来存储当前运行的结果
    results = []
    
    for n_initial in n_initial_values:
        if n_initial < last_n_initial:
            continue
        for repeat in range(n_repeats):
            if n_initial == last_n_initial and repeat <= last_repeat:
                continue
            print(f"\n=== 运行 n_initial={n_initial}, 重复={repeat+1}/{n_repeats} ===")
            
            # 主动学习（PM 方法）
            if 'PM' in methods_to_run and (last_method is None or last_method == 'PM'):
                print("--- 主动学习（PM 方法） ---")
                X_pool = generate_candidates(n_pool, lower_bounds, upper_bounds)
                X_train = generate_candidates(n_initial, lower_bounds, upper_bounds)
                u_train = non_test_function(X_train)
                for iter in range(n_iterations):
                    softmax_weights, mu, sigma, response_penalty = comsoftmax(
                        X_train, u_train, X_pool, target_value)
                    top_indices = np.argsort(-softmax_weights)[:n_preselect]
                    X_preselected = X_pool[top_indices]
                    space_filling = compute_space_filling_r_style(X_preselected, X_train)
                    space_filling_weights = space_filling_to_softmax(space_filling)
                    best_preselect_idx = np.argmax(space_filling_weights)
                    best_local_idx = top_indices[best_preselect_idx]
                    X_new = X_pool[best_local_idx].reshape(1, -1)
                    u_new = non_test_function(X_new)
                    X_train = np.vstack([X_train, X_new])
                    u_train = np.concatenate([u_train, u_new])
                    X_pool = np.delete(X_pool, best_local_idx, axis=0)
                RMSE_all_pm, RMSE_target_pm = evaluate_model(
                    X_train, u_train, Testx, Testy, target_value)
                results.append({
                    'method': 'PM',
                    'n_initial': n_initial,
                    'repeat': repeat + 1,
                    'RMSE_all': RMSE_all_pm,
                    'RMSE_target': RMSE_target_pm
                })
                save_progress(n_initial, repeat, 'PM', methods_to_run)
                print(f"PM 方法: RMSE_all={RMSE_all_pm:.4f}, RMSE_target={RMSE_target_pm:.4f}")
            
            # LHS 方法
            if 'LHS' in methods_to_run and (last_method is None or last_method == 'LHS'):
                print("--- LHS 方法 ---")
                n_lhs = n_initial + n_iterations
                X_lhs = generate_candidates(n_lhs, lower_bounds, upper_bounds)
                u_lhs = non_test_function(X_lhs)
                RMSE_all_lhs, RMSE_target_lhs = evaluate_model(
                    X_lhs, u_lhs, Testx, Testy, target_value)
                results.append({
                    'method': 'LHS',
                    'n_initial': n_initial,
                    'repeat': repeat + 1,
                    'RMSE_all': RMSE_all_lhs,
                    'RMSE_target': RMSE_target_lhs
                })
                save_progress(n_initial, repeat, 'LHS', methods_to_run)
                print(f"LHS 方法: RMSE_all={RMSE_all_lhs:.4f}, RMSE_target={RMSE_target_lhs:.4f}")
            
            # G-最优方法
            if 'G_opt' in methods_to_run and (last_method is None or last_method == 'G_opt'):
                print("--- G-最优方法 ---")
                X_train = generate_candidates(n_initial, lower_bounds, upper_bounds)
                u_train = non_test_function(X_train)
                X_candidates = generate_candidates(n_pool, lower_bounds, upper_bounds)
                for _ in range(n_iterations):
                    _, var_candidates = tgp_predict(X_train, u_train, X_candidates)
                    idx_max_var = np.argmax(var_candidates)
                    new_point = X_candidates[idx_max_var]
                    X_train = np.vstack([X_train, new_point])
                    u_train = np.append(u_train, non_test_function(new_point[np.newaxis, :]))
                    X_candidates = np.delete(X_candidates, idx_max_var, axis=0)
                RMSE_all_gopt, RMSE_target_gopt = evaluate_model(
                    X_train, u_train, Testx, Testy, target_value)
                results.append({
                    'method': 'G_opt',
                    'n_initial': n_initial,
                    'repeat': repeat + 1,
                    'RMSE_all': RMSE_all_gopt,
                    'RMSE_target': RMSE_target_gopt
                })
                save_progress(n_initial, repeat, 'G_opt', methods_to_run)
                print(f"G-最优方法: RMSE_all={RMSE_all_gopt:.4f}, RMSE_target={RMSE_target_gopt:.4f}")
            
            # D-最优方法
            if 'D_opt' in methods_to_run and (last_method is None or last_method == 'D_opt'):
                print("--- D-最优方法 ---")
                X_initial = generate_candidates(n_initial, lower_bounds, upper_bounds)
                u_initial = non_test_function(X_initial)
                X_candidates = generate_candidates(n_pool, lower_bounds, upper_bounds)
                X_additional = select_d_optimal_points(X_initial, X_candidates, n_iterations)
                u_additional = non_test_function(X_additional)
                X_train = np.vstack([X_initial, X_additional])
                u_train = np.concatenate([u_initial, u_additional])
                RMSE_all_dopt, RMSE_target_dopt = evaluate_model(
                    X_train, u_train, Testx, Testy, target_value)
                results.append({
                    'method': 'D_opt',
                    'n_initial': n_initial,
                    'repeat': repeat + 1,
                    'RMSE_all': RMSE_all_dopt,
                    'RMSE_target': RMSE_target_dopt
                })
                save_progress(n_initial, repeat, 'D_opt', methods_to_run)
                print(f"D-最优方法: RMSE_all={RMSE_all_dopt:.4f}, RMSE_target={RMSE_target_dopt:.4f}")
            
            # K-means 方法
            if 'K_means' in methods_to_run and (last_method is None or last_method == 'K_means'):
                print("--- K-means 方法 ---")
                X_train = generate_candidates(n_initial, lower_bounds, upper_bounds)
                u_train = non_test_function(X_train)
                X_candidates = generate_candidates(n_pool, lower_bounds, upper_bounds)
                kmeans = KMeans(n_clusters=n_iterations)
                kmeans.fit(X_candidates)
                new_pts = kmeans.cluster_centers_
                X_train = np.vstack([X_train, new_pts])
                u_train = np.append(u_train, non_test_function(new_pts))
                RMSE_all_kmeans, RMSE_target_kmeans = evaluate_model(
                    X_train, u_train, Testx, Testy, target_value)
                results.append({
                    'method': 'K_means',
                    'n_initial': n_initial,
                    'repeat': repeat + 1,
                    'RMSE_all': RMSE_all_kmeans,
                    'RMSE_target': RMSE_target_kmeans
                })
                save_progress(n_initial, repeat, 'K_means', methods_to_run)
                print(f"K-means 方法: RMSE_all={RMSE_all_kmeans:.4f}, RMSE_target={RMSE_target_kmeans:.4f}")
            
            # 立即将当前结果追加到 Excel 文件
            df = pd.DataFrame(results)
            append_to_excel(df, excel_filename)
            results = []  # 清空列表以便下次追加
    
    print("\n实验完成，所有结果已保存到 '五种方法结果.xlsx'")

if __name__ == "__main__":
    # 示例：只运行 PM 和 LHS 方法
    main(methods_to_run=[ 'K_means'])
