import numpy as np
from scipy.stats import multivariate_normal


######################################################
# 第 k 个模型的高斯分布密度函数
# 每 i 行表示第 i 个样本在各模型中的出现概率
# 返回一维列表
######################################################
def phi(X, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
    return norm.pdf(X)


######################################################
# E-step
######################################################
def getExpectation(X, mu, cov, alpha):
    N, K = X.shape[0], alpha.shape[0]
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"
    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = alpha[k] * phi(X, mu[k], cov[k])
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def update_membership_with_fairness(gamma, sensitive_groups, global_proportions, fairness_lambda):
    N, K = gamma.shape  # 数据点数量N，簇数量K
    for i in range(N):
        labels = np.argmax(gamma, axis=1)  # 当前的簇分配
        # fair_loss_value = fair_loss(labels, sensitive_groups, global_proportions)
        # 更新隶属度，考虑到公平损失和原始GMM的目标函数
        for j in range(K):
            cluster_points = sensitive_groups[labels == j]
            if len(cluster_points) == 0:
                continue  # 如果簇中没有数据点，跳过该簇
            cluster_counts = np.bincount(cluster_points, minlength=len(global_proportions))
            cluster_ratios = cluster_counts / cluster_counts.sum()
            ratio_difference = (cluster_ratios - global_proportions)
            if ratio_difference[sensitive_groups[i]] > 0:
                fair_loss_value = ratio_difference[sensitive_groups[i]] ** 2
                gamma[i, j] = gamma[i, j] * np.exp(-fairness_lambda * fair_loss_value)
    # 正则化隶属度
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    return gamma


######################################################
# M-step
######################################################
def maximize(X, gamma):
    N, D = X.shape
    K = gamma.shape[1]
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)
    for k in range(K):
        Nk = np.sum(gamma[:, k])
        mu[k, :] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / Nk
        diff = X - mu[k]
        cov_k = (diff.T @ (gamma[:, k][:, np.newaxis] * diff)) / Nk
        cov.append(cov_k)
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


######################################################
# 计算总体敏感属性比例
######################################################
def compute_global_proportions(sensitive_group):
    unique, counts = np.unique(sensitive_group, return_counts=True)
    proportions = counts / len(sensitive_group)
    return proportions


######################################################
# 计算每个簇中的敏感属性比例
######################################################
def compute_cluster_proportions(gamma, sensitive_group):
    N, K = gamma.shape
    unique_groups = np.unique(sensitive_group)
    num_groups = len(unique_groups)
    cluster_proportions = np.zeros((K, num_groups))

    for k in range(K):
        cluster_indices = np.where(np.argmax(gamma, axis=1) == k)[0]
        if len(cluster_indices) > 0:
            # 计算当前簇中每个群体的比例
            group_proportion = np.array([
                np.mean(sensitive_group[cluster_indices] == group)
                for group in unique_groups
            ])
        else:
            group_proportion = np.zeros(num_groups)  # 如果簇中没有样本，比例设置为0
        cluster_proportions[k, :] = group_proportion

    return cluster_proportions


######################################################
# 评估指标
######################################################
def compute_disappointment_score(cluster_proportions, global_proportions):
    K = cluster_proportions.shape[0]
    balance_scores = np.zeros(K)

    for k in range(K):
        cluster_diff = np.abs(cluster_proportions[k, :] - global_proportions)
        balance_scores[k] = np.sum(cluster_diff)
        # print("cluster", k+1, "：", cluster_proportions[k, :])
    return balance_scores


def evaluate_fairness(gamma, sensitive_group):
    cp = compute_cluster_proportions(gamma, sensitive_group)
    gp = compute_global_proportions(sensitive_group)
    # print("global_proportions：", gp)
    balance_proportion = compute_disappointment_score(cp, gp)
    balance_score = np.sum(balance_proportion)
    return balance_score


######################################################
# 初始化模型参数
# K 表示模型个数
######################################################
def init_params(X, K):
    N, D = X.shape
    gamma = np.zeros((N, K))
    random_indices = np.random.choice(X.shape[0], size=K, replace=False)
    mu = X[random_indices, :]
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return gamma, mu, cov, alpha


######################################################
# 高斯混合模型 EM 算法
# 给定样本矩阵 X，计算模型参数
# K 为模型个数
# max_iter 为迭代次数
######################################################
def FGMM(X, K, max_iter, sensitive_group, fairness_lambda=1.0):
    gamma, mu, cov, alpha = init_params(X, K)
    global_proportions = compute_global_proportions(sensitive_group)
    for i in range(max_iter):
        gamma = getExpectation(X, mu, cov, alpha)
        gamma = update_membership_with_fairness(gamma, sensitive_group, global_proportions, fairness_lambda)
        mu, cov, alpha = maximize(X, gamma)
    labels = np.argmax(gamma, axis=1)  # 获得每个样本的聚类标签
    return gamma, labels
