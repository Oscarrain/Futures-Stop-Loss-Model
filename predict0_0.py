import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import os

# 读取数据文件
data = pd.read_csv('ec2506_20240708.csv')

# 选择相关特征
features = data[['LastPrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'SellPrice01', 'BuyPrice01',
                 'SellVolume01', 'BuyVolume01', 'SellPrice02', 'BuyPrice02', 'SellVolume02', 'BuyVolume02',
                 'SellPrice03', 'BuyPrice03', 'SellVolume03', 'BuyVolume03', 'SellPrice04', 'BuyPrice04',
                 'SellVolume04', 'BuyVolume04', 'SellPrice05', 'BuyPrice05', 'SellVolume05', 'BuyVolume05']]

# 数据规范化
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# 构建数据集
window_size = 5
X = []
y = []

for i in range(len(scaled_df) - window_size - 5):
    X.append(scaled_df.iloc[i:(i+window_size)].values)
    y.append(scaled_df.iloc[i+window_size + 5]['LastPrice'])

X = np.array(X)
y = np.array(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 检查是否存在已保存的模型
model_path = 'lstm_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded from disk.")
else:
    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # 保存模型
    model.save(model_path)
    print("Model saved to disk.")


# 逐步预测测试集
y_pred = []
input_seq = X_test[0].copy()  # 初始化输入序列

for i in range(len(X_test)):
    pred = model.predict(input_seq[np.newaxis, :, :])
    y_pred.append(pred[0, 0])
    new_input = np.append(input_seq[1:], pred, axis=0)  # 更新输入序列
    input_seq = new_input

# 创建只包含最新价格的DataFrame用于逆归一化
last_price_scaler = MinMaxScaler()
last_price_scaler.fit_transform(features[['LastPrice']])

# 逆归一化
y_test_orig = last_price_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_orig = last_price_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))

# 计算误差指标
mse = mean_squared_error(y_test_orig, y_pred_orig)
print(f'Mean Squared Error: {mse}')

# 绘制实际值与预测值的折线图
plt.figure(figsize=(14, 7))
plt.plot(y_test_orig, label='Actual Price')
plt.plot(y_pred_orig, label='Predicted Price')
plt.title('Actual vs Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
# 实时预测函数
def predict_future_price(model, latest_data):
    latest_data_scaled = scaler.transform(latest_data)
    latest_data_scaled = np.array([latest_data_scaled])
    future_price_scaled = model.predict(latest_data_scaled)
    future_price = scaler.inverse_transform(future_price_scaled)
    return future_price[0][0]

# 实时预测循环(deprecated)
while False:
    # 读取最新的5条记录
    latest_data = scaled_df.iloc[-5:].values  # 你需要定义函数来获取最新的5条数据

    # 预测未来的最新价
    future_price = predict_future_price(model, latest_data)
    print(f"Predicted future price: {future_price}")

    # 等待250毫秒
    time.sleep(0.25)

# 根据另一个数据集，验证模型并绘图
def verify(filepath):
    data2 = pd.read_csv(filepath)
    features2 = data2[['LastPrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'SellPrice01', 'BuyPrice01',
                    'SellVolume01', 'BuyVolume01', 'SellPrice02', 'BuyPrice02', 'SellVolume02', 'BuyVolume02',
                    'SellPrice03', 'BuyPrice03', 'SellVolume03', 'BuyVolume03', 'SellPrice04', 'BuyPrice04',
                    'SellVolume04', 'BuyVolume04', 'SellPrice05', 'BuyPrice05', 'SellVolume05', 'BuyVolume05']]
    scaled_features2 = scaler.transform(features2)

    X2 = []
    y2 = []

    for i in range(len(scaled_features2) - window_size - 5):
        X2.append(scaled_features2[i:(i+window_size)].reshape(1, window_size, 24))
        y2.append(scaled_features2[i+window_size + 5][0])

    X2 = np.array(X2)
    y2 = np.array(y2)

    y2_pred = model.predict(X2)

    y2_test_orig = last_price_scaler.inverse_transform(y2.reshape(-1, 1))
    y2_pred_orig = last_price_scaler.inverse_transform(y2_pred)

    mse2 = mean_squared_error(y2_test_orig, y2_pred_orig)
    print(f'Mean Squared Error: {mse2}')

    plt.figure(figsize=(14, 7))
    plt.plot(y2_test_orig[:100], label='Actual Price')
    plt.plot(y2_pred_orig[:100], label='Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# verify('md/ec2506_2024-07-22.csv')

