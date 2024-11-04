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

window_size = 10
catagory = ['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']


def predict(model_path = 'lstm_model_1_0.h5'):
    # 检查是否存在已保存的模型
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model '{model_path}' loaded from disk.")
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

                X = []
                y = []

                for i in range(len(scaled_df) - window_size - 1):
                    X.append(scaled_df.iloc[i:(i+window_size)].values)
                    y.append(scaled_df.iloc[i+window_size + 1]['LastPrice'])

                X = np.array(X)
                y = np.array(y)

                # 划分训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


                # 训练模型
                model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

                # 保存模型
                model.save(model_path)
                print(f"Model saved to disk as '{model_path}'.")

    return model

def preprocess(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    else:
        data = pd.read_csv(file_path, names = ['InstrumentId', 'ExchangeId', 'Datetime', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'UpperLimitPrice', 'LowerLimitPrice', 'Volume', 'OpenInterest', 'BidPrice5', 'BidVolume5', 'BidPrice4', 'BidVolume4', 'BidPrice3', 'BidVolume3', 'BidPrice2', 'BidVolume2', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'AskPrice2', 'AskVolume2', 'AskPrice3', 'AskVolume3', 'AskPrice4', 'AskVolume4', 'AskPrice5', 'AskVolume5', 'LastPrice'])



    features = data[['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']]

    # 数据规范化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    x = scaled_df.iloc[-window_size:].values
    last_price = data.iloc[-1]['LastPrice']

    return x, last_price, scaler


def predict(model):

    file_path = input("Enter the file path: ")

    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    while True:

        x, last_price, scaler = preprocess(file_path)

        y = model.predict(x.reshape(1, window_size, x.shape[1]))[0][0]

        # # 逆缩放y和last_price
        # y = scaler.inverse_transform(np.concatenate([np.zeros((1, x.shape[1]-1)), y.reshape(-1, 1)], axis=1))[:, -1]
        # last_price = scaler.inverse_transform(np.concatenate([np.zeros((1, x.shape[1]-1)), last_price.reshape(-1, 1)], axis=1))[:, -1]

        print(f"Price change at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} by : {y - last_price}")

        time.sleep(0.249)

def main():
    # ask whether to train a new model
    train = input("Train a new model? (y/n): ")
    if train == 'y':
        model_path = input("Enter the model path, or press Enter to use the default model (end with .h5): ")
        if model_path == '':
            model_path = 'lstm_model_1_0.h5'
        model = predict(model_path)
    else:
        load = input("Load a default model? (y/n): ")
        if load == 'y':
            model = load_model('lstm_model_1_0.h5')
        else:
            model_path = input("Enter the model path, or press Enter to use the default model (end with .h5): ")
            if model_path == '':
                model_path = 'lstm_model_1_0.h5'
            model = load_model(model_path)
            print(f"Model '{model_path}' loaded from disk.")
    predict(model)

if __name__ == '__main__':
    main()