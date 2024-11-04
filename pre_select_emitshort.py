import os
import pandas as pd

# 如果文件内容少于10行，删除文件
def emit_short():
    for root, dirs, files in os.walk('data'):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            if len(df) < 15:
                os.remove(file_path)
                print(f"删除文件：{file_path}")

emit_short()