import pandas as pd
# 读取数据集
file_path = r"./dataset/original-diabetes.csv"
data = pd.read_csv(file_path, delimiter=',')
print(data.head())
# 检查是否存在 NaN 值
print(data.isna().sum())

# bank
# data = data[['balance', 'duration', 'marital']]
# data = data.dropna()
# data['marital'] = data['marital'].map({'single': 0, 'married': 1, 'divorced': 2})

# adult
# data.columns = data.columns.str.strip()
# data = data[['age','fnlwgt','hours-per-week','race']]
# data = data.dropna()
# data['race'] = data['race'].map({' White': 0, ' Black': 1,
# ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4}).astype(int)

# athlete
# data = data[['Age', 'Height', 'Weight', 'Sex']]
# data.replace([], pd.NA, inplace=True)
# data = data.dropna()
# # 去除所有重复行，只保留第一出现的那一行
# data = data.drop_duplicates()
# data['Sex'] = data['Sex'].map({'F': 0, 'M': 1}).astype(int)
# # 输出预处理后的数据集
# new_file_path = r"./dataset/bank.csv"
# data.to_csv(new_file_path, header=1, index=0)

# diabetes
data = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Outcome']]
data.replace([], pd.NA, inplace=True)
data = data.dropna()
data = data.drop_duplicates()
new_file_path = r"./dataset/diabetes.csv"
data.to_csv(new_file_path, header=1, index=0)
