import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

window_size = 10

catagory = ['LastPrice', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'AskPrice1', 'BidPrice1',
                                 'AskVolume1', 'BidVolume1', 'AskPrice2', 'BidPrice2', 'AskVolume2', 'BidVolume2',
                                 'AskPrice3', 'BidPrice3', 'AskVolume3', 'BidVolume3', 'AskPrice4', 'BidPrice4',
                                 'AskVolume4', 'BidVolume4', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']


def scaler_test():
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

                # 逆缩放y
                y = scaler.inverse_transform(np.concatenate([np.zeros((y.shape[0], len(catagory)-1)), y.reshape(-1, 1)], axis=1))[:, -1]

                # # 划分训练集和测试集
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # # 逆缩放预测值和真实值
                # y_test_scaled = scaler.inverse_transform(np.concatenate([np.zeros((y_test.shape[0], len(catagory)-1)), y_test.reshape(-1, 1)], axis=1))[:, -1]

                print('y')
                print(features['LastPrice'].values)
                print('y_scaled')
                print(y)

                break

scaler_test()