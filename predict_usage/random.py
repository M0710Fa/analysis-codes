# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import datetime

# データの読み込み
data = pd.read_csv("../../analysis-datas/src/2month_dd.csv", encoding="utf_8")

# 特徴量と目的変数を分割
X = data[['app', 'week', 'time']]
y = data['used']

# KFoldのインスタンスを作成
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# TimeSeriesSplitを用いる
tscv = TimeSeriesSplit(n_splits=5)

mse_list = []

# one-hotエンコーダーの初期化
encoder = OneHotEncoder(handle_unknown='ignore')

# K-fold交差検証
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 訓練データでone-hotエンコーダーをfitし、訓練データとテストデータの両方を変換
    X_train_encoded = encoder.fit_transform(X_train).toarray()
    X_test_encoded = encoder.transform(X_test).toarray()

    # ランダムフォレストモデルの訓練
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_encoded, y_train)

    # テストデータでの予測
    y_pred_rf = rf_model.predict(X_test_encoded)

    # Mean Squared Error (MSE) を計算
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mse_list.append(mse_rf)

# 各foldでのMSEと平均MSEを表示
for i, mse in enumerate(mse_list):
    print(f"MSE for fold {i+1}: {mse}")
    mae_msec  = datetime.timedelta(milliseconds=mse)
    print("mse（分）：{}".format(mae_msec.seconds/60))

m_mse = np.mean(mse_list)
print(f"Average MSE: {m_mse}")
m_mse_msec  = datetime.timedelta(milliseconds=mse)
print("m_mse（分）：{}".format(m_mse_msec.seconds/60))

# %%
