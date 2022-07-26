import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import make_scorer
import datetime

def rmse_score(y_true, y_pred):
    """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

df = pd.read_csv("../analysis-datas/src/usage_intest.csv")

print(df)

 # 説明変数
x = pd.get_dummies(df[["app","week","time"]],drop_first=True)
X = np.array(x)
# 目的変数
y = df["used"]
Y = np.array(y)

LR = LinearRegression()

kf = KFold(n_splits=4, shuffle=True, random_state=0)

# score_funcs = {
#     'rmse': make_scorer(rmse_score),
# }

# scores = cross_validate(LR, x, y, cv=kf, scoring=score_funcs)
# mean_rmse = scores['test_rmse'].mean()
# print('RMSE:', mean_rmse)
# rmse_msec  = datetime.timedelta(milliseconds=mean_rmse)
# print("rmse（分）：{}".format(rmse_msec.seconds/60))

# 各分割におけるスコア
# print('Cross-Validation scores: {}'.format(scores))

# 重回帰分析の結果を表示する
# print(result.summary())

# Xst = sm.add_constant(X_test)
# pred = result.predict(Xst)
# rmse = np.sqrt(mean_squared_error(Y_test,pred))

# rmse_msec  = datetime.timedelta(milliseconds=rmse)
# print("rmse（分）：{}".format(rmse_msec.seconds/60))