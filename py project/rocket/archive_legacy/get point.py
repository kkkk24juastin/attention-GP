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
from pyDOE2 import lhs
from sklearn.cluster import KMeans
import warnings
import logging

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

# 抑制rpy2包产生的日志和警告信息
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
        # 使用tryCatch处理TGP模型构建失败的情况
        r('model <- tryCatch(btgp(X_train, u_train, XX = X_candidates, verb=0), error=function(e) NULL)')
        if r('is.null(model)')[0]:
            print("警告: TGP模型构建失败，返回零均值和单位方差。")
            return np.zeros(X_candidates.shape[0]), np.ones(X_candidates.shape[0])
        r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')
        pred_mean = np.array(r("pred$mean"))
        pred_var = np.array(r("pred$var"))
        # 处理预测结果中的NaN或负值
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


# =============================================================================
# 3. 主动学习策略函数
# =============================================================================

def select_d_optimal_points(X_initial, X_candidates, n_additional):
    """为D-最优策略选择点"""
    X_initial_design = np.hstack([np.ones((X_initial.shape[0], 1)), X_initial])
    X_candidates_design = np.hstack([np.ones((X_candidates.shape[0], 1)), X_candidates])
    
    current_X = X_initial_design
    selected_indices_in_candidates = []
    available_indices = list(range(X_candidates_design.shape[0]))

    for _ in range(n_additional):
        max_increase = -np.inf
        best_candidate_index, best_local_index = -1, -1
        
        # 计算当前信息矩阵的行列式对数
        sign_current, current_logdet = np.linalg.slogdet(current_X.T @ current_X)
        if sign_current <= 0 or np.isinf(current_logdet): break
        
        # 遍历所有可用候选点，找到能最大化信息矩阵行列式的点
        for i, local_idx in enumerate(available_indices):
            new_X = np.vstack([current_X, X_candidates_design[local_idx]])
            sign_new, new_logdet = np.linalg.slogdet(new_X.T @ new_X)
            if sign_new <= 0: continue
            
            increase = new_logdet - current_logdet
            if increase > max_increase:
                max_increase, best_candidate_index, best_local_index = increase, local_idx, i
        
        if best_candidate_index != -1:
            selected_indices_in_candidates.append(best_candidate_index)
            current_X = np.vstack([current_X, X_candidates_design[best_candidate_index]])
            available_indices.pop(best_local_index)
            
    return X_candidates[selected_indices_in_candidates]


def run_g_opt_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations):
    print("--- 运行 G-最优 (G-opt) 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    pool_indices = np.arange(X_pool.shape[0])
    
    for i in range(n_iterations):
        X_remaining = X_pool[pool_indices]
        # 预测候选点的方差
        _, var_candidates = tgp_predict(X_train, u_train, X_remaining)
        # 选择方差最大的点
        idx_max_var_local = np.argmax(var_candidates)
        
        global_idx = pool_indices[idx_max_var_local]
        X_new = X_pool[global_idx:global_idx + 1, :]
        u_new = u_pool[global_idx:global_idx + 1]
        
        # 添加新点到训练集
        X_train = np.vstack([X_train, X_new])
        u_train = np.append(u_train, u_new)
        # 从池中移除已选择的点
        pool_indices = np.delete(pool_indices, idx_max_var_local)
        print(f"G-opt: 第 {i+1}/{n_iterations} 次迭代完成。")

    return X_train, u_train


def run_d_opt_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations):
    print("--- 运行 D-最优 (D-opt) 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    
    # 一次性选出所有D-最优的点
    X_additional = select_d_optimal_points(X_train, X_pool, n_iterations)
    print(f"D-opt: 已选出 {len(X_additional)}/{n_iterations} 个点。")
    
    # 找到这些点在原始池中的u值
    pool_map = {tuple(row): i for i, row in enumerate(X_pool)}
    u_additional_indices = [pool_map[tuple(point)] for point in X_additional]
    u_additional = u_pool[u_additional_indices]
    
    # 合并成最终的数据集
    X_final = np.vstack([X_train, X_additional])
    u_final = np.concatenate([u_train, u_additional])
    
    return X_final, u_final


def run_kmeans_method(X_train_orig, u_train_orig, X_pool, u_pool, n_iterations):
    print("--- 运行 K-means 方法 ---")
    X_train, u_train = X_train_orig.copy(), u_train_orig.copy()
    
    # 对候选池进行K-means聚类
    kmeans = KMeans(n_clusters=n_iterations, random_state=42, n_init=10)
    kmeans.fit(X_pool)
    
    # 将聚类中心作为新选择的点
    new_pts_X = kmeans.cluster_centers_
    u_additional = []
    # 对于每个聚类中心，找到候选池中离它最近的点的u值作为其u值
    for point in new_pts_X:
        distances_to_center = np.linalg.norm(X_pool - point, axis=1)
        closest_point_idx = np.argmin(distances_to_center)
        u_additional.append(u_pool[closest_point_idx])
    
    print(f"K-means: 已选出 {len(new_pts_X)}/{n_iterations} 个聚类中心点。")

    # 合并成最终的数据集
    X_final = np.vstack([X_train, new_pts_X])
    u_final = np.concatenate([u_train, np.array(u_additional)])
    
    return X_final, u_final


# =============================================================================
# 4. 主函数和实验流程
# =============================================================================
def main():
    # --- 实验参数设置 ---
    n_initial = 30          # 初始训练点数量
    n_iterations = 20       # 每种方法需要选择的新点数量
    n_pool_samples = 10000  # 候选池大小
    data_file = Path("rocketdata2.xlsx")
    output_file = Path("final_points_results.xlsx")
    lower_bounds = [0, 2, 0, 30, 1]
    upper_bounds = [20, 5, 1, 50, 2]

    # --- 数据准备 ---
    print(f"{'='*25} 数据准备 {'='*25}")
    # 1. 加载初始数据
    X_train_initial, u_train_initial = load_data(data_file, n_initial)
    print(f"已加载 {X_train_initial.shape[0]} 个初始点。")

    # 2. 生成候选池
    print("正在生成候选池 (LHS)...")
    X_pool = generate_candidates(n_pool_samples, lower_bounds, upper_bounds)
    print("正在为候选池生成预测值u (TGP)...")
    u_pool, _ = tgp_predict(X_train_initial, u_train_initial, X_pool)
    print(f"已生成包含 {X_pool.shape[0]} 个点的候选池。\n")

    # --- 运行各个方法 ---
    print(f"{'='*25} 运行主动学习方法 {'='*25}")
    # 运行G-opt方法
    X_g_opt, u_g_opt = run_g_opt_method(
        X_train_initial, u_train_initial, X_pool, u_pool, n_iterations
    )
    
    # 运行D-opt方法
    X_d_opt, u_d_opt = run_d_opt_method(
        X_train_initial, u_train_initial, X_pool, u_pool, n_iterations
    )
    
    # 运行K-means方法
    X_kmeans, u_kmeans = run_kmeans_method(
        X_train_initial, u_train_initial, X_pool, u_pool, n_iterations
    )

    # --- 保存结果 ---
    print(f"\n{'='*25} 保存结果 {'='*25}")
    with pd.ExcelWriter(output_file) as writer:
        # 准备G-opt结果
        df_g_opt = pd.DataFrame(X_g_opt, columns=[f'x{i+1}' for i in range(X_g_opt.shape[1])])
        df_g_opt['u'] = u_g_opt
        df_g_opt.to_excel(writer, sheet_name='G_opt', index=False)
        print(f"G-opt 方法的最终 {df_g_opt.shape[0]} 个点已保存到工作表 'G_opt'。")

        # 准备D-opt结果
        df_d_opt = pd.DataFrame(X_d_opt, columns=[f'x{i+1}' for i in range(X_d_opt.shape[1])])
        df_d_opt['u'] = u_d_opt
        df_d_opt.to_excel(writer, sheet_name='D_opt', index=False)
        print(f"D-opt 方法的最终 {df_d_opt.shape[0]} 个点已保存到工作表 'D_opt'。")

        # 准备K-means结果
        df_kmeans = pd.DataFrame(X_kmeans, columns=[f'x{i+1}' for i in range(X_kmeans.shape[1])])
        df_kmeans['u'] = u_kmeans
        df_kmeans.to_excel(writer, sheet_name='K_means', index=False)
        print(f"K-means 方法的最终 {df_kmeans.shape[0]} 个点已保存到工作表 'K_means'。")

    print(f"\n实验完成！所有结果已保存到文件: {output_file}")


if __name__ == "__main__":
    main()
