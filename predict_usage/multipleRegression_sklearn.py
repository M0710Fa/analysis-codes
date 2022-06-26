import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime

df = pd.read_csv("../analysis-datas/src/usage_intest.csv")

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
X = np.array(x)
# 目的変数
Y = np.array(df["used"])

# データの分割(訓練データとテストデータ)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

# 結果
print("【切片】:", model.intercept_)
print("【回帰係数】", *model.coef_,sep=",")
print("【決定係数(訓練)】:", model.score(X_train, Y_train))
# print("【決定係数(テスト)】:", model.score(X_test, Y_test))

# pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(Y_test, pred))
# print(pred)
# print(int(rmse))
# print(*np.array(pred,dtype="i8"),sep=',')
# rmse_msec  = datetime.timedelta(milliseconds=int(rmse))
# print("rmse（分）：{}".format(rmse_msec.seconds/60))