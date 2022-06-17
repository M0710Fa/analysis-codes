import pandas as pd
import numpy as np
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../analysis-datas/src/fixed_usage.csv')

# データの標準化
scaler = StandardScaler()
scaler.fit(df)
df_std = scaler.transfrom(np.array(df))
df_std = pd.DataFrame(df_std, columns= df.columns)

x = pd.get_dummies(df_std[["app","week","time"]],drop_first=True) # 説明変数
y = np.array(df_std['used']) # 目的変数

# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
X = sm.add_constant(df_std[x])

# 最小二乗法でモデル化
model = sm.OLS(y, X)
result = model.fit()

# 重回帰分析の結果を表示する
result.summary()


dt = pd.read_csv('../analysis-datas/tests/test-1day.csv')

x_test = pd.get_dummies(dt[["app","week","time"]] )
y_test = dt['used'] 

print(result.summary())
#print(result.predict())

#print("R2: {}".format(result.rsquared))

# fig = df.plot(x,y, figsize=(12,5), title='グラフ').figure()
# fig.savefig("img.png")