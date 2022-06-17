import pandas as pd
import numpy as np
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../analysis-datas/src/fixed_usage.csv')


x = pd.get_dummies(df[["app","week","time"]],drop_first=True) # 説明変数
y = df['used'] # 目的変数

# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
X = sm.add_constant(x)

# 最小二乗法でモデル化
model = sm.OLS(y, X)
result = model.fit()

# 重回帰分析の結果を表示する
result.summary()


dt = pd.read_csv('../analysis-datas/tests/test-1day.csv')

x_test = pd.get_dummies(dt[["app","week","time"]],drop_first=True)
y_test = dt['used'] 
X_test = sm.add_constant(x_test)

# print(result.summary())
print(result.predict(X))

print("R2: {}".format(result.rsquared))

# fig = df.plot(x,y, figsize=(12,5), title='グラフ').figure()
# fig.savefig("img.png")