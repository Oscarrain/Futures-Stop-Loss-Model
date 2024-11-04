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

    x = scaled_df.iloc[-window_size:].values
    last_price = scaled_df.iloc[-1]['LastPrice']

    return x, last_price, scaler


def predict(model_path, file_path):


    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model 'lstm_model_1_0.h5' loaded from disk.")
    else:
        print("No saved model found.")
        return

    while True:

        # print(f"Processing file: {file_path}")

        x, last_price, scaler = preprocess(file_path)

        y = model.predict(x.reshape(1, window_size, x.shape[1]))[0][0]

        # # 逆缩放y和last_price
        # y = scaler.inverse_transform(np.concatenate([np.zeros((1, x.shape[1]-1)), y.reshape(-1, 1)], axis=1))[:, -1]
        # last_price = scaler.inverse_transform(np.concatenate([np.zeros((1, x.shape[1]-1)), last_price.reshape(-1, 1)], axis=1))[:, -1]

        print(f"Price change at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} by : {(y - last_price):.3f}")

        time.sleep(0.24)
        

if __name__ == '__main__':
    model_path = 'lstm_model_1_0.h5'
    file_path = 'data/ec2410_2024-07-22.csv'
    predict(model_path, file_path)



