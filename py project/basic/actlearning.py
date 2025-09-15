import numpy as np
from pyDOE2 import lhs
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 激活 numpy 与 R 之间的自动转换
rpy2.robjects.numpy2ri.activate()

# 导入 R 中的 tgp 包
tgp = importr('tgp')
def non_test_function(X):
    x = X[:, 0]
    y = X[:, 1]
    step_function = np.where(x > 0, 1, 0)
    result = np.where(step_function == 1, 2 * np.sin(x) + y**2, x**2 + y**2)
    return result

# -------------------------------
# 生成拉丁超立方采样
def generate_lhs_samples(n_samples, n_dimensions, lower_bounds, upper_bounds):
    samples = lhs(n_dimensions, samples=n_samples)
    samples = samples * (np.array(upper_bounds) - np.array(lower_bounds)) + np.array(lower_bounds)
    return samples

# -------------------------------
# 将 numpy 数组转换为 R 矩阵
def to_r_matrix(np_array):
    n_rows, n_cols = np_array.shape
    return robjects.r.matrix(np_array, nrow=n_rows, ncol=n_cols)

# -------------------------------
# 调用 R 中的 btgp 函数进行建模与预测
def run_btgp(X_data, Z_data, XX_data, corr="exp", verb=0):

    X_r = to_r_matrix(X_data)
    Z_r = robjects.FloatVector(Z_data)
    XX_r = to_r_matrix(XX_data)
    result = tgp.btgp(X=X_r, Z=Z_r, XX=XX_r, corr=corr, verb=verb)
    return result

# -------------------------------
# 计算候选点与当前训练点之间的最小欧氏距离（空间填充度）
def compute_space_filling(X_pool, X_train):
    distances = np.sqrt(((X_pool[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
    min_distances = distances.min(axis=1)
    return min_distances

# -------------------------------
# 基于 BTGP 模型预测结果构造注意力得分，选择新采样点
def optimal_design_btgp(X_train, y_train, X_pool, target_value=1.5, n_select=1, epsilon=1e-8):

    result = run_btgp(X_train, y_train, X_pool, corr="exp", verb=0)
    mu = np.array(result.rx2("ZZ.mean"))
    sigma = np.array(result.rx2("ZZ.s2"))
    
    space_filling = compute_space_filling(X_pool, X_train)
    response_penalty = (mu - target_value)**2 + sigma

    norm_space = space_filling / (np.max(space_filling) + epsilon)
    norm_penalty = response_penalty / (np.max(response_penalty) + epsilon)
    
    score = np.log(norm_space + epsilon) - np.log(norm_penalty + epsilon)
    selected_indices = np.argsort(score)[-n_select:]
    
    return selected_indices, score, mu, sigma

# -------------------------------
# 主动学习过程（使用 R 的 btgp 实现 BTGP 模型）
def active_learning_btgp(n_initial=30, n_pool=1000, n_iterations=10, n_select=1,
                         lower_bounds=[-2, -2], upper_bounds=[2, 2], target_value=1.5):

    # 生成初始采样点
    X_initial = generate_lhs_samples(n_initial, 2, lower_bounds, upper_bounds)
    y_initial = non_test_function(X_initial)
    
    # 生成候选点池
    X_pool = generate_lhs_samples(n_pool, 2, lower_bounds, upper_bounds)
    
    # 初始化训练集
    X_train = X_initial.copy()
    y_train = y_initial.copy()
    history = []  # 记录每次迭代选择的候选点索引（相对于候选池）
    
    for iteration in range(n_iterations):
        print(f"==== 迭代 {iteration+1} ====")
        selected_indices, score, mu, sigma = optimal_design_btgp(X_train, y_train, X_pool,
                                                                  target_value=target_value,
                                                                  n_select=n_select)
        # 选中的新点及其响应
        X_new = X_pool[selected_indices, :]
        y_new = non_test_function(X_new)
        
        print("选中新点：")
        print(X_new)
        print("对应响应：")
        print(y_new)
        
        history.extend(selected_indices.tolist())
        X_train = np.vstack([X_train, X_new])
        y_train = np.concatenate([y_train, y_new])
        
        # 从候选池中删除已选点
        X_pool = np.delete(X_pool, selected_indices, axis=0)
    
    return X_train, y_train, history, X_initial, y_initial

# -------------------------------
# 绘制初始采样点、主动学习新增点以及响应1.5的等高线
def plot_results(X_initial, X_train_final, lower_bounds=[-2, -2], upper_bounds=[2, 2]):
    # 新增点：最终训练集中去除初始采样点的部分
    n_initial = X_initial.shape[0]
    X_added = X_train_final[n_initial:, :]
    
    # 构建网格，计算非平稳测试函数值
    x_grid = np.linspace(lower_bounds[0], upper_bounds[0], 200)
    y_grid = np.linspace(lower_bounds[1], upper_bounds[1], 200)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    Z_grid = non_test_function(grid_points).reshape(X_grid.shape)
    
    plt.figure(figsize=(8, 6))
    # 绘制等高线：只绘制响应值为 1.5 的等高线
    contour = plt.contour(X_grid, Y_grid, Z_grid, levels=[1.5], colors='blue', linewidths=2)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%.1f')
    
    # 绘制初始采样点（红色圆点）
    plt.scatter(X_initial[:, 0], X_initial[:, 1], c='red', marker='o', s=50, label='初始采样点')
    # 绘制主动学习新增点（绿色方块）
    if X_added.shape[0] > 0:
        plt.scatter(X_added[:, 0], X_added[:, 1], c='green', marker='s', s=50, label='主动学习增加点')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("初始采样点、主动学习新增点及响应=1.5等高线")
    plt.legend()
    plt.show()

# -------------------------------
# 主程序入口
if __name__ == "__main__":
    # 设置参数
    n_initial = 30
    n_pool = 1000
    n_iterations = 10
    n_select = 1          # 每次迭代选择 1 个新点
    lower_bounds = [-2, -2]
    upper_bounds = [2, 2]
    target_value = 1.5    # 目标响应值
    
    # 运行主动学习过程（使用 R 的 btgp 模型）
    X_train_final, y_train_final, history, X_initial, y_initial = active_learning_btgp(
        n_initial=n_initial,
        n_pool=n_pool,
        n_iterations=n_iterations,
        n_select=n_select,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        target_value=target_value
    )
    
    print("最终训练集大小:", X_train_final.shape)
    
    # 绘制结果：初始采样点、主动学习新增点及响应=1.5的等高线
    plot_results(X_initial, X_train_final, lower_bounds, upper_bounds)
