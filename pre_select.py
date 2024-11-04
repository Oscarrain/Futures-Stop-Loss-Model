import os
import pandas as pd
def pre_select():
    # 表头定义
    header = ["InstrumentId", "ExchangeId", "Datetime", "OpenPrice", "HighestPrice", "LowestPrice",
            "UpperLimitPrice", "LowerLimitPrice", "Volume", "OpenInterest", "BidPrice5", "BidVolume5",
            "BidPrice4", "BidVolume4", "BidPrice3", "BidVolume3", "BidPrice2", "BidVolume2", "BidPrice1",
            "BidVolume1", "AskPrice1", "AskVolume1", "AskPrice2", "AskVolume2", "AskPrice3", "AskVolume3",
            "AskPrice4", "AskVolume4", "AskPrice5", "AskVolume5", "LastPrice"]

    # 创建data文件夹
    if not os.path.exists('data'):
        os.makedirs('data')

    # 遍历md文件夹及其子文件夹
    for root, dirs, files in os.walk('md'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                
                # 检查文件名长度是否小于8
                if len(file_name) < 20:
                    df = pd.read_csv(file_path, header=None)
                    print(f"处理文件：{file}")
                    
                    # 检查是否有表头
                    if df.iloc[0].isnull().all():
                        df.columns = header
                    else:
                        df.columns = header
                    
                    # 保存到data文件夹
                    new_file_path = os.path.join('data', file)
                    df.to_csv(new_file_path, index=False)

    print("文件处理完成。")

def try_one():
    # 表头定义
    header = ["InstrumentId", "ExchangeId", "Datetime", "OpenPrice", "HighestPrice", "LowestPrice",
            "UpperLimitPrice", "LowerLimitPrice", "Volume", "OpenInterest", "BidPrice5", "BidVolume5",
            "BidPrice4", "BidVolume4", "BidPrice3", "BidVolume3", "BidPrice2", "BidVolume2", "BidPrice1",
            "BidVolume1", "AskPrice1", "AskVolume1", "AskPrice2", "AskVolume2", "AskPrice3", "AskVolume3",
            "AskPrice4", "AskVolume4", "AskPrice5", "AskVolume5", "LastPrice"]

    # 创建data文件夹
    if not os.path.exists('data'):
        os.makedirs('data')

    counter = 0

    # 遍历md文件夹及其子文件夹
    for root, dirs, files in os.walk('md'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                
                # 检查文件名长度是否小于8
                if len(file_name) < 20:
                    df = pd.read_csv(file_path, header=None)
                    counter += 1
                    print(f"处理文件：{file}")
                    
                    # 检查是否有表头
                    if df.iloc[0].isnull().all():
                        df.columns = header
                        # print(f"文件{file}没有表头")
                        # print(df.head())
                    else:
                        df.columns = header
                        # print(f"文件{file}有表头")
                        # print(df.head())
                        # print(df.columns)

                    if counter >= 5:
                        break
                    
                    # 保存到data文件夹
                    new_file_path = os.path.join('data', file)
                    df.to_csv(new_file_path, index=False)
        
    # # 读取文件ec2506_20240708.csv
    # data1 = pd.read_csv('ec2506_20240708.csv')
    # print(data1.columns)

def verify():
    # 读取文件md/ec2506_2024-07-22.csv
    data2 = pd.read_csv('md/ec2506_2024-07-22.csv')
    print(data2.columns)

if __name__ == '__main__':
    verify()
