import numpy as np
import matplotlib.pyplot as plt
n, p, miss = 100, 7, 5
# test_matrix = np.array([np.nan, 0, 3, 7, 2, 6, 5, 1, 2, np.nan, np.nan, 5]).reshape(4, 3)
# 生成随机的正态分布并取出一些值
def generate_normal_matrix(n, p, missing_num, initial_mean=13):
    # 生成第一列的数据，均值为 initial_mean
    matrix = np.random.normal(loc=initial_mean, scale=0.5, size=(n, 1))

    # 生成剩余列的数据，每一列的均值比前一列的均值多1
    for i in range(1, p):
        mean = initial_mean + i
        column = np.random.normal(loc=mean, scale=0.5, size=(n, 1))
        matrix = np.concatenate((matrix, column), axis=1)

    # 随机丢失值的数量
    indices_to_remove = np.random.choice(range(n * p), size=missing_num, replace=False)
    matrix_flat = matrix.flatten()  # 将矩阵展平为一维数组
    matrix_flat[indices_to_remove] = np.nan  # 将选定的索引位置的值设为空
    # 重新形成矩阵
    matrix = matrix_flat.reshape((n, p))
    return matrix


# 找到NaN的索引
def nan_index(matrix):
    matrix = np.array(matrix)
    missing_indices = np.argwhere(np.isnan(matrix))
    return missing_indices


# 计算均值矩阵
def calculate_mean_matrix(matrix):
    mean_matrix = np.nanmean(matrix,axis=0)
    return mean_matrix


# 将空值填充
def padding_function(null_matrix):
    for i in range(p):
        if np.isnan(null_matrix[:, i]).any():
            null_matrix[:, i][np.isnan(null_matrix[:, i])] = calculate_mean_matrix(null_matrix)[i]
    full_matrix = null_matrix
    return full_matrix


# 计算协方差矩阵
def covariance_matrix_calculate(matrix):
    transposed_matrix = matrix.T  # 对原矩阵进行转置，使变量作为行
    cov_matrix = np.cov(transposed_matrix, rowvar=True, ddof=0)  # ddof=1返回的是无偏估计，否则为极大似然估计，rowvar=True意味每一行是一个变量
    return cov_matrix


# EM算法补充缺失值
def em_algorithm(matrix, max_iter=100, tol=1e-9):
    missing_indices = nan_index(matrix)
    # 初始填充缺失值
    filled_matrix = padding_function(matrix.copy())
    missing_values_history = {}
    for idx in missing_indices:
        missing_values_history[tuple(idx)] = []
    for iteration in range(max_iter):
        #防止修改影响
        old_filled_matrix = filled_matrix.copy()
        mean_matrix = calculate_mean_matrix(filled_matrix)
        cov_matrix = covariance_matrix_calculate(filled_matrix)
        for row, col in missing_indices:
            #求出每个缺失值所在行的缺失情况
            observed_indices = ~np.isnan(matrix[row])
            #求出行中非缺失值的元素
            observed_values = filled_matrix[row, observed_indices]
            #求出非缺失值的均值
            observed_mean = mean_matrix[observed_indices]
            #分割求出子矩阵
            observed_cov = cov_matrix[np.ix_(observed_indices, observed_indices)]#便于提取子矩阵
            #求出交叉矩阵，例如均值μ1和μ3有关，则交叉矩阵为σ13
            cross_cov = cov_matrix[col, observed_indices]
            #修改值
            missing_value_estimate = mean_matrix[col] + cross_cov @ np.linalg.inv(observed_cov) @ (
                        observed_values - observed_mean)
            filled_matrix[row, col] = missing_value_estimate
            missing_values_history[(row, col)].append(missing_value_estimate)
        #求出误差
        if np.linalg.norm(filled_matrix - old_filled_matrix) < tol:
            break

    return filled_matrix,missing_values_history

# 测试
filled_matrix,missing_values_history = em_algorithm(generate_normal_matrix(n,p,miss))
print("Matrix after EM algorithm:")
print(filled_matrix)

plt.figure(figsize=(10, 6))
for (row, col), history in missing_values_history.items():
    plt.plot(history, label=f'Missing value at ({row}, {col})')

plt.xlabel('Iteration')
plt.ylabel('Estimated Value')
plt.title('Convergence of Missing Values in EM Algorithm')
plt.legend()
plt.show()


