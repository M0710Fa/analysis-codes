import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime

df = pd.read_csv("../analysis-datas/src/usage_intest.csv")

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=False)
X = np.array(x)
# 目的変数
y = df["used"]
Y = np.array(y)

# データの分割(訓練データとテストデータ)
<<<<<<< HEAD
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

=======
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
>>>>>>> d72a306f8361578bcaa3032510a54e49c34a9aac


# 結果
print("【切片】:", int(model.intercept_))
print(*np.array(model.coef_,dtype="i8"),sep=',')
# print("【決定係数(テスト)】:", model.score(X_test, Y_test))

<<<<<<< HEAD
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, pred))
print(pred)
print(int(rmse))
print(*np.array(pred,dtype="i8"),sep=',')
=======
pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(pred)
print(int(rmse))
# print(*np.array(pred,dtype="i8"),sep=',')
>>>>>>> d72a306f8361578bcaa3032510a54e49c34a9aac
rmse_msec  = datetime.timedelta(milliseconds=int(rmse))
print("rmse（分）：{}".format(rmse_msec.seconds/60))