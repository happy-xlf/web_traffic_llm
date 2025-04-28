import requests
import os
from openai import OpenAI
import json
from tqdm import tqdm
import re
import concurrent.futures
import threading
import pandas as pd
from datetime import datetime

def reshape_datetime(input_file, out_file):
    df = pd.read_csv(input_file)
     # 转换时间格式
    df['Time'] = df['Time'].apply(lambda x: 
                                    datetime.strptime(x, '%m/%d/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
                                )
    
    # 保存转换后的数据
    df.to_csv(out_file, index=False)
    print(f"转换成功! 结果已保存到 {out_file}")

input_file = './ec_data_set.csv'
output_file = './ec_data.csv'
reshape_datetime(input_file, output_file)
