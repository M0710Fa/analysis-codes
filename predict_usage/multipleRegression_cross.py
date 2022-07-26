import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../analysis-datas/src/cross_1/sepdata.csv")

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
X = np.array(x)
# 目的変数
y = df["used"]
Y = np.array(y)



# 最小二乗法でモデル化
model = LinearRegression()
model.fit(X, Y)


dt = pd.read_csv('../analysis-datas/src/cross_1/test.csv')

x_test = pd.get_dummies(dt[["app","week","time"]],drop_first=True)
X_test = np.array(x_test)
y_test = dt['used'] 
Y_test = np.array(y_test)


pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test,pred))
print(rmse)
print(pred)
print(*np.array(pred,dtype="i8"),sep=',')
rmse_msec  = datetime.timedelta(milliseconds=rmse)
print("rmse（分）：{}".format(rmse_msec.seconds/60))