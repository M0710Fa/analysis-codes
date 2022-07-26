import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import numpy as np
import datetime

df = pd.read_csv("../analysis-datas/src/2month_6-7.csv", encoding="utf_8")

 # 説明変数
x = pd.get_dummies(df[["week","time"]],drop_first=False)
X = np.array(x)
# 目的変数
y = df["used"]
Y = np.array(y)

# データの分割(訓練データとテストデータ)
<<<<<<< HEAD
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
=======
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=0)
>>>>>>> d72a306f8361578bcaa3032510a54e49c34a9aac

# 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
Xsm = sm.add_constant(X)

# 最小二乗法でモデル化
model = sm.OLS(Y, Xsm)
result = model.fit()

# 重回帰分析の結果を表示する
print(result.summary())

Xst = sm.add_constant(X_test)
pred = result.predict(Xst)
rmse = np.sqrt(mean_squared_error(Y_test,pred))
<<<<<<< HEAD
print(rmse)
print(pred)
print(*np.array(pred,dtype="i8"),sep=',')
=======
# print(rmse)
# print(pred)
# print(*np.array(pred,dtype="i8"),sep=',')
>>>>>>> d72a306f8361578bcaa3032510a54e49c34a9aac
rmse_msec  = datetime.timedelta(milliseconds=rmse)
print("rmse（分）：{}".format(rmse_msec.seconds/60))

#print(*result.params, sep=",")
<<<<<<< HEAD
#print(*np.array(result.params,dtype="i8"),sep=',')
=======
# print(*np.array(result.params,dtype="i8"),sep=',')
>>>>>>> d72a306f8361578bcaa3032510a54e49c34a9aac


# dt = pd.read_csv('../analysis-datas/tests/test-1day.csv')

# x_test = pd.get_dummies(dt[["app","week","time"]],drop_first=True)
# y_test = dt['used'] 
# X_test = sm.add_constant(x_test)

# # print(result.summary())
# print(result.predict(X))

# print("R2: {}".format(result.rsquared))

# fig = df.plot(x,y, figsize=(12,5), title='グラフ').figure()
# fig.savefig("img.png")