import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('macd/ec2412_2024-07-29.csv')

# 计算MACD(deprecated)
def deprecated_calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['signal_line'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal_line']
    return df

# 计算MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['dif'] = df['ema_short'] - df['ema_long']
    df['dea'] = df['dif'].ewm(span=signal_window, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
    return df

df = calculate_macd(df)

# 计算MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['dif'] = df['ema_short'] - df['ema_long']
    df['dea'] = df['dif'].ewm(span=signal_window, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
    df['macd_hist_cumulative'] = df['macd'].cumsum()
    return df

df = calculate_macd(df)

# 绘制MACD图像
def plot_macd(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Close Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df['date'], df['macd_hist_cumulative'], label='Cumulative MACD Histogram', color='purple')
    ax2.set_title('Cumulative MACD Histogram Area')
    ax2.set_ylabel('Cumulative Area')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.legend()

    ax3.plot(df['date'], df['dif'], label='DIF', color='blue')
    ax3.plot(df['date'], df['dea'], label='DEA', color='red')
    ax3.bar(df['date'], df['macd'], label='MACD Histogram', color='gray')
    ax3.set_title('MACD')
    ax3.set_ylabel('MACD Value')
    ax3.legend()

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_macd(df)
