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

all_tests = []
window_size = 10
catagory = ['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']


def predict():
    # 检查是否存在已保存的模型
    model_path = 'lstm_model_1_0.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model 'lstm_model_1_0.h5' loaded from disk.")
    else:
        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(window_size, len(catagory))))
        model.add(LSTM(50))
        model.add(Dense(1))

        # 编译模型
        model.compile(optimizer='adam', loss='mean_squared_error')

    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                print(f"处理文件：{file}")
                # press enter to continue
                # input("Press Enter to continue...")
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                data = pd.read_csv(file_path)
                # 选择相关特征
                # InstrumentId,ExchangeId,Datetime,OpenPrice,HighestPrice,LowestPrice,UpperLimitPrice,LowerLimitPrice,Volume,OpenInterest,BidPrice5,BidVolume5,BidPrice4,BidVolume4,BidPrice3,BidVolume3,BidPrice2,BidVolume2,BidPrice1,BidVolume1,AskPrice1,AskVolume1,AskPrice2,AskVolume2,AskPrice3,AskVolume3,AskPrice4,AskVolume4,AskPrice5,AskVolume5,LastPrice
                features = data[['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']]

                # 数据规范化
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(features)
                scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

                # print(scaled_df.head())

                X = []
                y = []

                for i in range(len(scaled_df) - window_size - 1):
                    X.append(scaled_df.iloc[i:(i+window_size)].values)
                    y.append(scaled_df.iloc[i+window_size + 1]['LastPrice'])

                X = np.array(X)
                y = np.array(y)

                # 划分训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                all_tests.append((X_test, y_test, file_name, scaler))


                # 训练模型
                model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

                # 保存模型
                model.save(model_path)
                print("Model 'lstm_model_1_0.h5' saved to disk.")

    # save all_tests as a file
    np.save('all_tests.npy', np.array(all_tests, dtype=object))

# 校验测试集
def test():
    # 检查是否存在已保存的模型
    model_path = 'lstm_model_1_0.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model 'lstm_model_1_0.h5' loaded from disk.")
    else:
        print("No saved model found.")
        return

    # 加载测试数据
    if os.path.exists('all_tests.npy'):
        all_tests = np.load('all_tests.npy', allow_pickle=True)
    else:
        print("No test data found.")
        return

    # 进行预测并计算误差
    for (X_test, y_test, file_name, scaler) in all_tests:
        print(f"Testing file: {file_name}")

        # 确保形状匹配
        if X_test.shape[1:] != (window_size, len(catagory)):
            print(f"Shape mismatch for file: {file_name}")
            continue

        # 预测
        y_pred = model.predict(X_test)
        
        # 逆缩放预测值和真实值
        y_test_scaled = scaler.inverse_transform(np.concatenate([np.zeros((y_test.shape[0], len(catagory)-1)), y_test.reshape(-1, 1)], axis=1))[:, -1]
        y_pred_scaled = scaler.inverse_transform(np.concatenate([np.zeros((y_pred.shape[0], len(catagory)-1)), y_pred], axis=1))[:, -1]

        # 计算误差
        mse = np.mean((y_test_scaled - y_pred_scaled) ** 2)
        print(f"Mean Squared Error for {file_name}: {mse}")

        # 遍历每个时刻
        for i in range(1, len(y_test_scaled)):
            # # 绘制每个时刻的图
            # plt.figure(figsize=(12, 6))
            # plt.plot(range(i+1), y_test_scaled[:i+1], color='red', label='Real Last Price')
            # plt.plot([i-1, i], [y_test_scaled[i-1], y_pred_scaled[i]], color='blue', label='Predicted Last Price')
            # plt.title(f'Last Price Prediction at time {i} - {file_name}')
            # plt.xlabel('Time')
            # plt.ylabel('Last Price')
            # plt.legend()
        
            # # 保存图片到pics文件夹
            # if not os.path.exists('pics'):
            #     os.makedirs('pics')
            # plt.savefig(f'pics/{file_name}_{i}.png')
            # plt.close()

            # 输出该时刻的涨跌幅度
            print(f"Price change at time {i}: {(y_pred_scaled[i] - y_test_scaled[i-1]):.3f}")



def view_test():
    all_tests = np.load('all_tests.npy', allow_pickle=True)
    print(all_tests[0])
    X_test, y_test, file_name, scaler = all_tests[0]
    print(X_test.shape, y_test.shape, file_name, scaler)
    print(X_test[0], y_test[0], file_name, scaler)

def preprocess_data(data):
    features = data[['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']]
    scaler = MinMaxScaler()
    scaled_features = scaler.transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    # X仅保留最后window_size行
    X = scaled_df.iloc[-window_size:].values


def predict_future_price(model, latest_data):
    latest_data_scaled = scaler.transform(latest_data)
    latest_data_scaled = np.array([latest_data_scaled])
    future_price_scaled = model.predict(latest_data_scaled)
    future_price = scaler.inverse_transform(future_price_scaled)
    return future_price[0][0]



test()