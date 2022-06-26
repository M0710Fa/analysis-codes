import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import numpy as np
import datetime

def rmse_score(y_true, y_pred):
    """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

df = pd.read_csv("../analysis-datas/src/usage_intest.csv")

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
X = np.array(x)
# 目的変数
Y = np.array(df["used"])

# データの分割(訓練データとテストデータ)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
Xsm = sm.add_constant(X_train)

# 最小二乗法でモデル化
model = sm.OLS(Y_train, Xsm)
result = model.fit()

# 重回帰分析の結果を表示する
print(result.summary())

Xst = sm.add_constant(X_test)
pred = result.predict(Xst)
rmse = np.sqrt(mean_squared_error(Y_test,pred))
print(pred)
print(int(rmse))
rmse_msec  = datetime.timedelta(milliseconds=rmse)
print("rmse（分）：{}".format(rmse_msec.seconds/60))