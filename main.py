import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from gmm_group_fairness import FGMC, evaluate_fairness
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.mixture import GaussianMixture


file_path = r"./dataset/bank.csv"
data = pd.read_csv(file_path)
X = data[['balance', 'duration']].values
sensitive_groups = data['marital'].values

# file_path = r"./dataset/adult.csv"
# data = pd.read_csv(file_path)
# X = data[['age', 'fnlwgt']].values
# sensitive_groups = data['race'].values

# file_path = r"./dataset/athlete.csv"
# data = pd.read_csv(file_path)
# X = data[['Age', 'Height', 'Weight']].values
# sensitive_groups = data['Sex'].values

# file_path = r"./dataset/diabetes.csv"
# data = pd.read_csv(file_path)
# X = data[['Glucose', 'BloodPressure', 'SkinThickness']].values
# sensitive_groups = data['Outcome'].values

# 取一部分数据点
num_samples = 700
random_indices = np.random.choice(X.shape[0], num_samples, replace=False)
X = X[random_indices, :]
sensitive_groups = sensitive_groups[random_indices]
# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

K = 8                                                                                             
# GMM using Standard EM Algorithm
gmm = GaussianMixture(K)
gmm.fit(X)
labels1 = gmm.predict(X)
gamma1 = gmm.predict_proba(X)
# 计算轮廓系数
silhouette_score_GMC = silhouette_score(X, labels1)
print(f"silhouette_score_GMM: {silhouette_score_GMC}")
disappointment_score_GMC, equal_score_GMC = evaluate_fairness(gamma1, sensitive_groups)
print(f"disappointment_score_GMM: {disappointment_score_GMC}")
print(f"equal_score_GMC: {equal_score_GMC}")
print("--------------------------------------")

# GMM using self EM Algorithm
gamma2, labels2 = FGMC(X, K, 100, sensitive_groups, fairness_lambda=1.0)
# 计算轮廓系数
silhouette_score_FGMC = silhouette_score(X, labels2)
print(f"silhouette_score_FGMM: {silhouette_score_FGMC}")
disappointment_score_FGMC, equal_score_FGMC = evaluate_fairness(gamma2, sensitive_groups)
print(f"disappointment_score_FGMM: {disappointment_score_FGMC}")
print(f"equal_score_FGMC: {equal_score_FGMC}")
