import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../analysis-datas/src/fixed_usage.csv")

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
x1 = [["app"]]
x2 = [["week"]]
x3 = [["time"]]

# 目的変数
y = df[["used"]]

scaler = StandardScaler()
scaler.fit(x)
x_std = scaler.transform(x)
scaler.fit(y)
y_std = scaler.transform(y)

model = LinearRegression()
model.fit(x_std, y_std)


dt = pd.read_csv("../analysis-datas/tests/test-1day.csv")
test_x = pd.get_dummies(dt[["app","week","time"]],drop_first=True)
test_y = dt[["used"]]

pred = scaler.inverse_transform(model.predict(test_x))
# true_y = test_y.values
# rmse = np.sqrt(mean_squared_error(true_y, pred))

# print("重回帰モデル　coef {}".format(model.coef_))
# print("予測の精度　{}".format(model.score(x_std, y_std)))
# print(x)
# print("rmse_result　{}".format(rmse))