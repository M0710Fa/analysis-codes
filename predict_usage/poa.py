# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# データの読み込み
data = pd.read_csv("../../analysis-datas/src/2month_dd.csv", encoding="utf_8")

# 'app'をダミー変数に変換
data = pd.get_dummies(data, columns=['app','week','time'])

# 特徴量と目的変数を分割
X = data.drop('used', axis=1)
y = data['used']

# TimeSeriesSplitを用いる
tscv = TimeSeriesSplit(n_splits=5)

# 交差検証を行う
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    


    # ポアソン回帰モデルの適用
    poisson_model = sm.GLM(y_train, sm.add_constant(X_train.toarray), family=sm.families.Poisson())
    poisson_results = poisson_model.fit()

    # 予測
    y_pred = poisson_results.predict(sm.add_constant(X_test))

    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse}")
