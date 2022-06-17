import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, scale

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
print(model.coef_)
print(model.score(x_std, y_std))

dt = pd.read_csv("../analysis-datas/tests/test-1day.csv")
test_x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
test_y = df[["used"]]

pred = scaler.inverse_transform(model.predict(test_x))
print(pred)