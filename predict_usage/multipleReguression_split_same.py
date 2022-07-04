import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import datetime

def statsmodels(x, xt, y, yt):
    xsm = sm.add_constant(x)

    # 最小二乗法でモデル化
    model = sm.OLS(y, xsm)
    result = model.fit()
    print(result.summary())
    print("////////////////")
    print(*np.array(result.params,dtype="f4"),sep=',')

def sklearn(x, xt, y, yt):
    model = LinearRegression()
    model.fit(x, y)
    print("【切片】:", int(model.intercept_))
    print(*np.array(model.coef_,dtype="f4"),sep=',')


df = pd.read_csv("../analysis-datas/src/fixed_usage.csv")

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
X = np.array(x)
# 目的変数
y =df["used"]
Y = np.array(y)

# データの分割(訓練データとテストデータ)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

statsmodels(x, X_test, y, Y_test)
sklearn(x, X_test, y, Y_test)

# print("//////データスプリット////////////")
statsmodels(X_train, X_test, Y_train, Y_test)
sklearn(X_train, X_test, Y_train, Y_test)
