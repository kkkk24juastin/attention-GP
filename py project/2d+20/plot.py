import numpy as np  
from scipy.spatial.distance import cdist  
import torch  
from pyDOE2 import lhs  
import rpy2.robjects as robjects  
from rpy2.robjects import numpy2ri  
numpy2ri.activate()  
import matplotlib.pyplot as plt  
import matplotlib  
matplotlib.use("TkAgg")  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
from sklearn.cluster import KMeans
import pandas as pd  

# 设置全局参数  
plt.rcParams['font.size'] = 15  # 设置全局字体大小  
plt.rcParams['lines.markersize'] = 15  # 设置散点图中点的大小  
plt.rcParams['axes.titlesize'] = 15  # 设置 axes 标题的字体大小  
plt.rcParams['axes.labelsize'] = 15  # 设置 axes 标签的字体大小  
plt.rcParams['xtick.labelsize'] = 15  # 设置 x 轴刻度标签的字体大小  
plt.rcParams['ytick.labelsize'] = 15  # 设置 y 轴刻度标签的字体大小  
plt.rcParams['legend.fontsize'] = 15  # 设置图例字体大小  
plt.rcParams['figure.titlesize'] = 15  # 设置 figure 标题的字体大小  

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

# 保存采样结果到xlsx文件的函数
def save_sampling_results(method_results, filename="sampling_results.xlsx"):
    """
    将采样结果保存到Excel文件中
    
    参数:
    method_results: 包含各种方法采样结果的字典
    filename: 输出文件名
    """
    with pd.ExcelWriter(filename) as writer:
        for method_name, result in method_results.items():
            # 合并初始样本和额外样本
            if result['initial'].size > 0 and result['additional'].size > 0:
                all_samples = np.vstack([result['initial'], result['additional']])
                sample_types = ['initial'] * len(result['initial']) + ['additional'] * len(result['additional'])
            elif result['initial'].size > 0:
                all_samples = result['initial']
                sample_types = ['initial'] * len(result['initial'])
            elif result['additional'].size > 0:
                all_samples = result['additional']
                sample_types = ['additional'] * len(result['additional'])
            else:
                continue  # 跳过空结果
            
            # 创建DataFrame
            df = pd.DataFrame({
                'x1': all_samples[:, 0],
                'x2': all_samples[:, 1],
                'sample_type': sample_types,
                'method': method_name,
                'label': result['label'],
                'color': result['color'],
                'marker': result['marker']
            })
            
            # 计算函数值
            df['function_value'] = non_test_function(all_samples)
            
            # 保存到Excel的不同工作表
            df.to_excel(writer, sheet_name=method_name, index=False)
    
    print(f"采样结果已保存到 {filename}")

# 各种采样方法的实现
def run_pm_method(X_initial, X_pool, n_iterations, n_preselect, target_value):
    """主动学习（PM方法）"""
    X_pool = X_pool.copy()  # 复制候选池避免修改原始数据
    X_train = X_initial.copy()
    u_train = non_test_function(X_train)
    X_active = np.empty((0, 2))
    
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
        X_active = np.vstack([X_active, X_new])
        X_pool = np.delete(X_pool, best_local_idx, axis=0)
    
    return X_initial, X_active

def run_lhs_method(X_initial, n_iterations, lower_bounds, upper_bounds):
    """LHS方法"""
    # LHS方法应该一次性生成所有样本点以保证空间填充特性
    n_total = len(X_initial) + n_iterations  # 总共需要的样本数
    X_all_lhs = generate_candidates(n_total, lower_bounds, upper_bounds)
    
    # LHS不需要区分初始和额外样本，所有点都作为一个整体
    X_empty = np.empty((0, 2))  # 空的初始样本
    
    return X_empty, X_all_lhs

def run_g_optimal_method(X_initial, X_candidates, n_iterations):
    """G-最优方法"""
    X_candidates = X_candidates.copy()  # 复制候选池避免修改原始数据
    X_train = X_initial.copy()
    u_train = non_test_function(X_train)
    X_active = np.empty((0, 2))
    
    for _ in range(n_iterations):
        _, var_candidates = tgp_predict(X_train, u_train, X_candidates)
        idx_max_var = np.argmax(var_candidates)
        new_point = X_candidates[idx_max_var].reshape(1, -1)
        X_train = np.vstack([X_train, new_point])
        u_train = np.append(u_train, non_test_function(new_point))
        X_active = np.vstack([X_active, new_point])
        X_candidates = np.delete(X_candidates, idx_max_var, axis=0)
    
    return X_initial, X_active

def run_d_optimal_method(X_initial, X_candidates, n_iterations):
    """D-最优方法"""
    X_additional = select_d_optimal_points(X_initial, X_candidates, n_iterations)
    return X_initial, X_additional

def run_cvt_method(X_initial, X_candidates, n_iterations):
    """CVT方法（使用K-means实现）"""
    kmeans = KMeans(n_clusters=n_iterations)
    kmeans.fit(X_candidates)
    X_additional = kmeans.cluster_centers_
    return X_initial, X_additional

# 测试函数 - 只生成函数图像  
def test_function_visualization():  
    # 绘制等高线图  
    xx = np.linspace(-2, 2, 400)  
    yy = np.linspace(-2, 2, 400)  
    Xg, Yg = np.meshgrid(xx, yy)  
    Z = non_test_function(np.vstack([Xg.ravel(), Yg.ravel()]).T).reshape(Xg.shape)  
    plt.figure(figsize=(8, 4.5))  
    cf = plt.contourf(Xg, Yg, Z, levels=50, cmap='jet')  
    contour_lines = plt.contour(Xg, Yg, Z, colors='black', levels=10, linewidths=0.5)  
    target_contour = plt.contour(Xg, Yg, Z, levels=[1.5], colors='red', linestyles='dashed', linewidths=2)  
    plt.xlabel('$x_1$')  
    plt.ylabel('$x_2$', rotation=-360)  
    plt.title('测试函数总体图像')  
    
    # 添加图例  
    import matplotlib.lines as mlines  
    black_line = mlines.Line2D([], [], color='black', linewidth=0.5, label='等高线')  
    red_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label='目标值等高线 (f=1.5)')  
    plt.legend(handles=[black_line, red_line], bbox_to_anchor=(0.98, 0.98), loc='upper right', prop={'size': 10}, frameon=True, fancybox=True, shadow=True, handlelength=1.0, handletextpad=0.3)  
    
    plt.colorbar(cf)  
    plt.savefig('test_function_visualization.pdf', format='pdf', bbox_inches='tight', dpi=300)  
    plt.show()  

# 采样主函数（只负责采样和保存数据）  
def run_sampling_methods(methods_to_run=['PM'], filename="sampling_results.xlsx"):  
    """
    运行各种采样方法并保存结果到xlsx文件
    
    参数：
    methods_to_run: 要运行的方法列表，可选: ['PM', 'LHS', 'G_opt', 'D_opt', 'CVT']
    filename: 保存结果的文件名
    """
    lower_bounds = [-2, -2]  
    upper_bounds = [2, 2]  
    target_value = 1.5  
    n_pool = 3000  
    n_iterations = 20  
    n_preselect = 40  
    n_initial = 20  # 初始训练样本数量  

    # 生成统一的初始样本和候选池，所有方法都使用这些相同的数据
    print("生成统一的初始样本和候选池...")
    X_initial_unified = generate_candidates(n_initial, lower_bounds, upper_bounds)
    X_candidates_unified = generate_candidates(n_pool, lower_bounds, upper_bounds)
    
    # 准备存储各种方法的结果
    method_results = {}
    
    # 运行选定的方法
    if 'PM' in methods_to_run:
        print("运行PM方法...")
        X_initial_pm, X_active_pm = run_pm_method(X_initial_unified, X_candidates_unified, 
                                                 n_iterations, n_preselect, target_value)
        method_results['PM'] = {'initial': X_initial_pm, 'additional': X_active_pm, 
                               'label': 'PM', 'color': 'red', 'marker': 'D'}
    
    if 'LHS' in methods_to_run:
        print("运行LHS方法...")
        X_initial_lhs, X_additional_lhs = run_lhs_method(X_initial_unified, n_iterations, 
                                                        lower_bounds, upper_bounds)
        method_results['LHS'] = {'initial': X_initial_lhs, 'additional': X_additional_lhs,
                               'label': 'LHS', 'color': 'blue', 'marker': 's'}
    
    if 'G_opt' in methods_to_run:
        print("运行G-最优方法...")
        X_initial_gopt, X_additional_gopt = run_g_optimal_method(X_initial_unified, X_candidates_unified, 
                                                               n_iterations)
        method_results['G_opt'] = {'initial': X_initial_gopt, 'additional': X_additional_gopt,
                                 'label': 'G-optimal', 'color': 'green', 'marker': '^'}
    
    if 'D_opt' in methods_to_run:
        print("运行D-最优方法...")
        X_initial_dopt, X_additional_dopt = run_d_optimal_method(X_initial_unified, X_candidates_unified, 
                                                               n_iterations)
        method_results['D_opt'] = {'initial': X_initial_dopt, 'additional': X_additional_dopt,
                                 'label': 'D-optimal', 'color': 'orange', 'marker': 'v'}
    
    if 'CVT' in methods_to_run:
        print("运行CVT方法...")
        X_initial_cvt, X_additional_cvt = run_cvt_method(X_initial_unified, X_candidates_unified, 
                                                        n_iterations)
        method_results['CVT'] = {'initial': X_initial_cvt, 'additional': X_additional_cvt,
                               'label': 'CVT', 'color': 'cyan', 'marker': 'o'}

    # 保存结果到xlsx文件
    save_sampling_results(method_results, filename)
    
    print(f"采样完成，共运行了 {len(methods_to_run)} 种方法")
    return method_results

# 专门的绘图函数
def plot_sampling_results(filename="sampling_results.xlsx", save_plot=True):
    """
    从xlsx文件读取采样数据并绘制等高线图和采样点分布
    
    参数:
    filename: 包含采样结果的xlsx文件名
    save_plot: 是否保存图片到文件
    """
    try:
        # 读取Excel文件中的所有工作表
        excel_file = pd.ExcelFile(filename)
        method_data = {}
        
        print(f"从 {filename} 加载采样数据...")
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(filename, sheet_name=sheet_name)
            method_data[sheet_name] = df
        
        if not method_data:
            print("错误：没有找到采样数据")
            return
        
        # 绘制等高线图
        print("绘制结果图...")
        xx = np.linspace(-2, 2, 400)  
        yy = np.linspace(-2, 2, 400)  
        Xg, Yg = np.meshgrid(xx, yy)  
        Z = non_test_function(np.vstack([Xg.ravel(), Yg.ravel()]).T).reshape(Xg.shape)  
        
        plt.figure(figsize=(8, 4.5))  
        cf = plt.contourf(Xg, Yg, Z, levels=50, cmap='jet')  
        plt.contour(Xg, Yg, Z, colors='black', levels=10, linewidths=0.5)  
        plt.contour(Xg, Yg, Z, levels=[1.5], colors='red', linestyles='dashed', linewidths=2)
        
        # 绘制各种方法的采样点
        initial_plotted = False
        methods_used = []
        
        for method_name, df in method_data.items():
            if df.empty:
                continue
            
            methods_used.append(method_name)
            
            # 分离初始样本和额外样本
            initial_samples = df[df['sample_type'] == 'initial']
            additional_samples = df[df['sample_type'] == 'additional']
            
            # 获取方法的绘图属性
            if not df.empty:
                color = df.iloc[0]['color']
                marker = df.iloc[0]['marker']
                label = df.iloc[0]['label']
            
            # 只绘制一次初始样本点（所有方法使用相同的初始点）
            if not initial_plotted and not initial_samples.empty:
                plt.scatter(initial_samples['x1'], initial_samples['x2'], c='purple', s=100, 
                           label='Initial', marker='o', 
                           edgecolors='black', linewidth=1)
                initial_plotted = True
            
            # 绘制该方法添加的样本点
            if not additional_samples.empty:
                plt.scatter(additional_samples['x1'], additional_samples['x2'], 
                           c=color, s=120, marker=marker,
                           label=f"{label}", 
                           edgecolors='black', linewidth=1)
        
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$', rotation=-360)
        plt.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', prop={'size': 9}, 
                   frameon=True, fancybox=True, shadow=True, markerscale=0.6, 
                   handlelength=1.0, handletextpad=0.3)
        plt.colorbar(cf)
        
        # 根据方法保存图片
        if save_plot:
            plot_filename = f'sampling_methods_comparison_{"_".join(methods_used)}.pdf'
            plt.savefig(plot_filename, format='pdf', bbox_inches='tight', dpi=300)
            print(f"图片已保存到 {plot_filename}")
        
        plt.show()
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
        print("请先运行采样方法生成数据")
    except Exception as e:
        print(f"绘图时发生错误：{e}")

if __name__ == "__main__":  
    # 选择运行模式：
    
    # 1. 只查看函数图像（不进行采样）
    # test_function_visualization()
    
    # 2. 运行采样方法并保存数据到xlsx文件
    print("=== 执行采样方法 ===")
    methods_to_run = ['PM', 'LHS', 'G_opt', 'D_opt', 'CVT']  # 可以修改这里选择要运行的方法
    filename = "sampling_results.xlsx"
    
    # # 运行采样
    # run_sampling_methods(methods_to_run=methods_to_run, filename=filename)
    
    # 3. 从xlsx文件读取数据并绘图
    print("\n=== 读取数据并绘图 ===")
    plot_sampling_results(filename=filename, save_plot=True)
    
    print("\n=== 运行完成 ===")
    print("可用的采样方法:")
    print("- PM: Active learning with attention mechanism") 
    print("- LHS: Latin Hypercube Sampling")
    print("- G_opt: G-optimal design") 
    print("- D_opt: D-optimal design")
    print("- CVT: Centroidal Voronoi Tessellation")
    print(f"\n采样数据已保存到: {filename}")
    print("图片已保存为PDF格式")
    
    # 如果只想运行部分方法，可以修改methods_to_run，例如：
    # methods_to_run = ['PM', 'LHS']  # 只运行PM和LHS方法
    # run_sampling_methods(methods_to_run=methods_to_run, filename="pm_lhs_results.xlsx")
    # plot_sampling_results(filename="pm_lhs_results.xlsx", save_plot=True)  
