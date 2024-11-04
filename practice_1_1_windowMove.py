import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

import time


window_size = 10

def preprocess(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    else:
        data = pd.read_csv(file_path)
        # print headers
        # print(data.head())



    features = data[['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']]

    # 数据规范化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    # x = scaled_df.iloc[-window_size:].values
    # last_price = scaled_df.iloc[-1]['LastPrice']

    X = []
    last_price = []
    real_y = []

    for i in range(len(scaled_df) - window_size - 1):
        X.append(scaled_df.iloc[i:(i+window_size)].values)
        last_price.append(scaled_df.iloc[i+window_size]['LastPrice'])
        real_y.append(scaled_df.iloc[i+window_size + 1]['LastPrice'])

    return X, last_price, scaler, real_y


def predict(model_path, file_path):


    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model 'lstm_model_1_0.h5' loaded from disk.")
    else:
        print("No saved model found.")
        return

    # print(f"Processing file: {file_path}")

    X, last_price, scaler, real_y = preprocess(file_path)


    # y = model.predict(x.reshape(1, window_size, x.shape[1]))[0][0]
    y = model.predict(np.array(X))


    # # 逆缩放y和last_price
    # y = scaler.inverse_transform(np.concatenate([np.zeros((1, x.shape[1]-1)), y.reshape(-1, 1)], axis=1))[:, -1]
    # last_price = scaler.inverse_transform(np.concatenate([np.zeros((1, x.shape[1]-1)), last_price.reshape(-1, 1)], axis=1))[:, -1]

    # 数据统计
    count = 0
    right_direction = 0
    eighty_percent = 0
    fifty_percent = 0

    for i in range(len(y)):
        real = real_y[i] - last_price[i]
        estimated = y[i][0] - last_price[i]
        print(f"Price change at {i} by : {(real):.3f}, estimated: {(estimated):.3f}")

        if real != 0:
            count += 1
        if (real > 0 and estimated > 0) or (real < 0 and estimated < 0):
            right_direction += 1
            rate = abs(estimated) / abs(real)
            if rate >= 0.5:
                fifty_percent += 1
                if rate >= 0.8:
                    eighty_percent += 1

        
        # time.sleep(0.24)

    print(f"Total: {count}")
    print(f"right direction: {right_direction}, accuracy: {right_direction/count:.3f}")
    print(f"80% accuracy: {eighty_percent/count:.3f}")
    print(f"50% accuracy: {fifty_percent/count:.3f}")
        

if __name__ == '__main__':
    model_path = 'lstm_model_1_0.h5'
    file_path = input("Enter the file path: ")
    if not os.path.exists(file_path):
        default_file_path = 'data/ec2410_2024-07-22.csv'
        default = input(f"File {file_path} not found, using default file {default_file_path}. Continue? (y/n): ")
        if default == 'y':
            file_path = default_file_path
        else:
            exit()
    predict(model_path, file_path)



