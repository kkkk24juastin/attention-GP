import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from sklearn.cluster import KMeans

# ### Data Loading
def load_data(file_path):
    """Load features and responses from an Excel file."""
    data = pd.read_excel(file_path)
    X = data.iloc[:, 0:3].values  # First 3 columns as features
    u = data.iloc[:, 3].values    # 4th column as response
    return X, u

# Load datasets
X_TEST, u_test = load_data("data.xlsx")
X_train_init, u_train_init = load_data("acttrain.xlsx")
X_pool, u_pool = load_data("actpool.xlsx")
X_lhs, u_lhs = load_data("lhstrain.xlsx")

# ### TGP Prediction
def tgp_predict(X_train, u_train, X_candidates):
    """Predict mean and variance using the tgp package in R."""
    r = robjects.r
    r.assign("X_train", X_train)
    r.assign("u_train", u_train)
    r.assign("X_candidates", X_candidates)
    r('library(tgp)')
    r('model <- btgp(X_train, u_train, XX = X_candidates)')
    r('pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)')
    return np.array(r("pred$mean")), np.array(r("pred$var"))

# ### Space-Filling and Softmax Functions
def compute_space_filling_r_style(X_candidates, X_train):
    """Compute normalized minimum distances from candidates to training points."""
    n_candidates = X_candidates.shape[0]
    space_filling = np.zeros(n_candidates)
    for j in range(n_candidates):
        distances = np.sqrt(np.sum((X_candidates[j] - X_train) ** 2, axis=1))
        space_filling[j] = np.min(distances)
    return space_filling / np.max(space_filling)

def space_filling_to_softmax(space_filling):
    """Convert space-filling scores to softmax weights."""
    space_filling = np.log(space_filling)
    space_filling_tensor = torch.tensor(space_filling, dtype=torch.float32)
    return torch.softmax(space_filling_tensor, dim=0).numpy()

# ### Candidate Features for Active Learning
def compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value):
    """Compute prediction mean, variance, and normalized response penalty."""
    mu, sigma = tgp_predict(X_train, u_train, X_candidates)
    response_penalty = (mu - target_value) ** 2 + sigma
    return mu, sigma, response_penalty / np.max(response_penalty)

def comsoftmax(X_train, u_train, X_candidates, target_value):
    """Compute softmax weights based on response penalty."""
    mu, sigma, response_penalty = compute_candidate_features_tgp(X_train, u_train, X_candidates, target_value)
    response_penalty_neg = -np.log(response_penalty)
    log_norm_penalty_tensor = torch.tensor(response_penalty_neg, dtype=torch.float32)
    softmaxpointres = torch.softmax(log_norm_penalty_tensor, dim=0)
    return softmaxpointres.numpy(), mu, sigma, response_penalty

# ### D-Optimality Functions
def d_optimality(X):
    """Compute the determinant of X^T X for D-optimality."""
    return np.linalg.det(X.T @ X)

def select_d_optimal_points(X_initial, X_candidates, n_additional):
    """Select points maximizing the increase in the information matrix determinant."""
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
    return selected_indices

# ### Point Selection Methods
def active_learning(X_train, u_train, X_pool, u_pool, n_iterations, n_preselect, target_value):
    """Active learning with response penalty and space-filling criteria."""
    X_train_new = X_train.copy()
    u_train_new = u_train.copy()
    remaining_indices = list(range(X_pool.shape[0]))
    for _ in range(n_iterations):
        X_candidates = X_pool[remaining_indices]
        softmax_weights, _, _, _ = comsoftmax(X_train_new, u_train_new, X_candidates, target_value)
        top_indices = np.argsort(-softmax_weights)[:n_preselect]
        X_preselected = X_candidates[top_indices]
        space_filling = compute_space_filling_r_style(X_preselected, X_train_new)
        space_filling_weights = space_filling_to_softmax(space_filling)
        best_preselect_idx = np.argmax(space_filling_weights)
        best_local_idx = top_indices[best_preselect_idx]
        best_global_index = remaining_indices[best_local_idx]
        X_new = X_pool[best_global_index].reshape(1, -1)
        u_new = u_pool[best_global_index].reshape(1,)
        X_train_new = np.vstack([X_train_new, X_new])
        u_train_new = np.concatenate([u_train_new, u_new])
        remaining_indices.remove(best_global_index)
    return X_train_new, u_train_new

def g_optimal(X_train, u_train, X_pool, u_pool, n_iterations):
    """G-optimality: Select points maximizing prediction variance."""
    X_train_new = X_train.copy()
    u_train_new = u_train.copy()
    remaining_indices = list(range(X_pool.shape[0]))
    for _ in range(n_iterations):
        X_candidates = X_pool[remaining_indices]
        _, var_candidates = tgp_predict(X_train_new, u_train_new, X_candidates)
        idx_max_var = np.argmax(var_candidates)
        best_global_index = remaining_indices[idx_max_var]
        X_new = X_pool[best_global_index].reshape(1, -1)
        u_new = u_pool[best_global_index].reshape(1,)
        X_train_new = np.vstack([X_train_new, X_new])
        u_train_new = np.concatenate([u_train_new, u_new])
        remaining_indices.remove(best_global_index)
    return X_train_new, u_train_new

def d_optimal(X_train, u_train, X_pool, u_pool, n_iterations):
    """D-optimality: Select points maximizing the information matrix determinant."""
    X_train_new = X_train.copy()
    u_train_new = u_train.copy()
    remaining_indices = list(range(X_pool.shape[0]))
    for _ in range(n_iterations):
        X_candidates = X_pool[remaining_indices]
        selected_indices = select_d_optimal_points(X_train_new, X_candidates, 1)
        best_index = selected_indices[0]
        best_global_index = remaining_indices[best_index]
        X_new = X_pool[best_global_index].reshape(1, -1)
        u_new = u_pool[best_global_index].reshape(1,)
        X_train_new = np.vstack([X_train_new, X_new])
        u_train_new = np.concatenate([u_train_new, u_new])
        remaining_indices.remove(best_global_index)
    return X_train_new, u_train_new

def k_means_selection(X_train, u_train, X_pool, u_pool, n_iterations):
    """K-means: Select points closest to cluster centers."""
    kmeans = KMeans(n_clusters=n_iterations)
    kmeans.fit(X_pool)
    centers = kmeans.cluster_centers_
    distances = cdist(centers, X_pool)
    closest_indices = np.argmin(distances, axis=1)
    X_new = X_pool[closest_indices]
    u_new = u_pool[closest_indices]
    X_train_new = np.vstack([X_train, X_new])
    u_train_new = np.concatenate([u_train, u_new])
    return X_train_new, u_train_new

# ### Evaluation Function
def evaluate_model(X_train, u_train, X_test, u_test, target_range=(25, 35)):
    """Evaluate the model on the test set and compute RMSE for all and target range."""
    mu, _ = tgp_predict(X_train, u_train, X_test)
    rmse_all = np.sqrt(np.mean((mu - u_test) ** 2))
    mask = (u_test >= target_range[0]) & (u_test <= target_range[1])
    if mask.any():
        rmse_target = np.sqrt(np.mean((mu[mask] - u_test[mask]) ** 2))
    else:
        rmse_target = np.nan
    return rmse_all, rmse_target

# ### Main Function
def main(methods_to_run=['PM', 'LHS', 'G_opt', 'D_opt', 'K_means'], n_iterations=30, n_preselect=50, target_value=30, n_repeats=30):
    results = []
    for method in methods_to_run:
        for repeat in range(n_repeats):
            print(f"\n=== 运行 {method} 方法, 重复 {repeat+1}/{n_repeats} ===")
            if method == 'PM':
                X_train, u_train = active_learning(X_train_init, u_train_init, X_pool, u_pool, n_iterations, n_preselect, target_value)
            elif method == 'G_opt':
                X_train, u_train = g_optimal(X_train_init, u_train_init, X_pool, u_pool, n_iterations)
            elif method == 'D_opt':
                X_train, u_train = d_optimal(X_train_init, u_train_init, X_pool, u_pool, n_iterations)
            elif method == 'K_means':
                X_train, u_train = k_means_selection(X_train_init, u_train_init, X_pool, u_pool, n_iterations)
            elif method == 'LHS':
                X_train, u_train = X_lhs, u_lhs  # Assuming X_lhs and u_lhs are pre-loaded with appropriate size
            else:
                raise ValueError(f"Unknown method: {method}")
            
            rmse_all, rmse_target = evaluate_model(X_train, u_train, X_TEST, u_test)
            results.append({
                'method': method,
                'repeat': repeat + 1,
                'RMSE_all': rmse_all,
                'RMSE_target': rmse_target
            })
            print(f"{method} 方法: RMSE_all={rmse_all:.4f}, RMSE_target={rmse_target:.4f}")
    
    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel('results.xlsx', index=False)
    print("\n实验完成，所有结果已保存到 'results.xlsx'")    

if __name__ == "__main__":
    # Example: Run only PM and LHS methods
    main(methods_to_run=['G_opt', 'D_opt', 'K_means'])
