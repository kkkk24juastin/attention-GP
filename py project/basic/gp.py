import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from pyDOE2 import lhs
import matplotlib.pyplot as plt

def generate_lhs_samples(n_samples, n_dimensions, lower_bounds, upper_bounds):
    samples = lhs(n_dimensions, samples=n_samples)
    for i in range(n_dimensions):
        samples[:, i] = samples[:, i] * (upper_bounds[i] - lower_bounds[i]) + lower_bounds[i]
    return samples

n_samples = 25    # 采样点数量
n_dimensions = 2  # 维度数
lower_bounds = [-2,2]  # 各维度下界
upper_bounds = [-2,2]  # 各维度上界

def run_optimization(X): 
    X = np.asarray(X)
    Z = np.zeros_like(X, dtype=float)
    mask = X <= 10
    Z[mask] = (np.sin(2 * np.pi * X[mask] / 7) * 
               np.exp(-0.1 * X[mask]) + 
               0.3 * np.cos(3 * np.pi * X[mask] / 5)) 
    mask = X > 10
    Z[mask] = -0.5 + 0.2 * X[mask] + 0.1 * np.sin(X[mask])
    return Z

# 导入R的TGP包
tgp = importr('tgp')
X = generate_lhs_samples(n_samples, n_dimensions, lower_bounds, upper_bounds)
Z = run_optimization(X)
XX = generate_lhs_samples(100, n_dimensions, lower_bounds, upper_bounds)

# 将NumPy数组转换为R对象
X_r = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
Z_r = robjects.FloatVector(Z)
XX_r = robjects.r.matrix(XX, nrow=XX.shape[0], ncol=XX.shape[1])

# 调用btgp函数
result = tgp.btgp(X_r, Z_r, XX_r)

print(result.rx('trees'))

