import requests
import os
from openai import OpenAI
import json
from tqdm import tqdm
import re
import concurrent.futures
import threading
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from domain_prompt import base_prompt, analyse_abnormal_prompt, analyse_pred_abnormal_prompt

client = OpenAI(
    api_key="xxx",
    base_url="http://0.0.0.0:8666/v1",
)

def get_res(content):
    completion = client.chat.completions.create(
        model="DeepSeek-R1-32B",
        messages=[
            {"role": "user", "content": content}
        ],
    )
    predict_content = completion.choices[0].message.content

    return predict_content

def get_data(file_path):
    data = pd.read_csv(file_path)
    new_data = []
    for time, traffic in data.values:
        new_data.append(time + " - " + str(int(traffic)))

    return new_data

def gen_forecast_time(data):
    time_steps = [10, 15, 20, 25]
    forecast_steps = [1, 3, 5, 7]
    json_data = []
    # 基于既定的规则获取
    for time_step in tqdm(time_steps, total=len(time_steps), desc="generate"):
        for forecast_step in forecast_steps:
            for i in range(0,len(data) - time_step - forecast_step + 1, time_step//2):
                ori_time_str = "\n".join(data[i:i+time_step])
                ans_data_str = "\n".join(data[i+time_step:i+time_step+forecast_step])
                k = random.randint(0, len(base_prompt)-1)
                query_prompt = base_prompt[k]
                json_data.append({
                    "instruction": query_prompt.replace("{{原始流量序列}}", ori_time_str).replace("{{预测流量序列}}", str(ans_data_str)),
                    "input": "",
                    "output": ""
                })
    # 随机产生time_step和forecast_step
    # 随机产生time_step和forecast_step
    random_samples = len(json_data)  # Define how many random samples you want to generate
    random_json_data = []
    for _ in tqdm(range(random_samples), desc="generate random patterns"):

        while True:
            # Randomly select a time_step between 10 and 600
            random_time_step = random.randint(10, 30)
            # Randomly select a forecast_step between 1 and 600
            random_forecast_step = random.randint(1, 10)
            if abs(random_time_step - random_forecast_step) < random_time_step:
                break
        
        # Make sure we have enough data for both steps
        if random_time_step + random_forecast_step <= len(data):
            # Randomly select a starting point
            max_start = len(data) - random_time_step - random_forecast_step
            if max_start > 0:
                start_idx = random.randint(0, max_start)
                
                ori_time_str = "\n".join(data[start_idx:start_idx+random_time_step])
                ans_data_str = "\n".join(data[start_idx+random_time_step:start_idx+random_time_step+random_forecast_step])
                
                k = random.randint(0, len(base_prompt)-1)
                query_prompt = base_prompt[k]
                random_json_data.append({
                    "instruction": query_prompt.replace("{{原始流量序列}}", ori_time_str).replace("{{预测流量序列}}", str(ans_data_str)),
                    "input": "",
                    "output": ""
                })
    # random dataset and split train_dataset and test_dataset

    # 1.Shuffle the dataset randomly
    random.shuffle(json_data)
    random.shuffle(random_json_data)
    
    # 2.Split into training and test sets (80% train, 20% test)
    split_idx = int(len(json_data) * 0.9)
    train_data = json_data[:split_idx]
    test_data = json_data[split_idx:]

    random_split_idx = int(len(random_json_data) * 0.9)
    random_train_data = random_json_data[:random_split_idx]
    random_test_data = random_json_data[random_split_idx:]

    train_data = train_data + random_train_data
    test_data = test_data + random_test_data

    return train_data, test_data 



def get_forecast_dataset():
    file_path = "./data/ec_data.csv"
    data = get_data(file_path)
    train_data, test_data = gen_forecast_time(data)

    file_path = "./data/uk_data.csv"
    data = get_data(file_path)
    train_data2, test_data2 = gen_forecast_time(data)

    ec_test_data = test_data[:len(test_data)//5]
    uk_test_data = test_data2[:len(test_data2)//5]
    
    with open("./dataset/test/short_ec_forecast_test.json", "w", encoding="utf-8") as f:
        json.dump(ec_test_data, f, ensure_ascii=False, indent=4)

    with open("./dataset/test/short_uk_forecast_test.json", "w", encoding="utf-8") as f:
        json.dump(uk_test_data, f, ensure_ascii=False, indent=4)


def abnormal_data(file_path):
   
    # 获取文件夹下的所有文件名
    all_files = os.listdir(file_path)
    
    # Filter for CSV files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    all_train_data = []
    all_test_data = []
    all_train_data_3 = []
    all_test_data_3 = []
    
    # Read each CSV file
    for csv_file in csv_files:
        full_path = os.path.join(file_path, csv_file)
        # try:
        # Read the data
        df = pd.read_csv(full_path)
        # Process the data similar to get_data()
        data = []
        anomaly_labels = []
        for time, traffic, is_anomaly in df.values:
            data.append("第"+str(int(time))+"个流量值" + ": " + str(traffic))
            anomaly_labels.append(int(is_anomaly))
        
        time_steps = [10, 20, 30, 40, 50, 60]
        json_data = []
        json_data_3 = []
        # 基于既定的规则获取
        for time_step in tqdm(time_steps, total=len(time_steps), desc="generate"):
            for i in range(time_step, len(data) - time_step + 1, time_step):
                old_time_str = "\n".join(data[i-time_step:i])
                ori_time_str = "\n".join(data[i:i+time_step])
                
                # 找到anomaly_labels中为1的下标，并提取出对应的ori_time_abnormal，如果没有则返回[]
                anomaly_indices = [idx for idx in range(i, i+time_step) if idx < len(anomaly_labels) and anomaly_labels[idx] == 1]
                
                # Extract the anomalous data points based on indices
                anomalous_data = []
                for idx in anomaly_indices:
                    anomalous_data.append(data[idx])
                
                # Determine if this window contains anomalies
                has_anomalies = len(anomalous_data) > 0
                
                # Generate the appropriate output based on whether anomalies were found
                if has_anomalies:
                    # Format the anomalous points for output
                    anomaly_points_str = "\n".join(anomalous_data)
                
                    json_data.append({
                        "instruction": analyse_abnormal_prompt.replace("{{原始流量序列}}", ori_time_str).replace("{{异常点流量}}", anomaly_points_str),
                        "input": "",
                        "output": ""
                    })
                    json_data_3.append({
                        "instruction": analyse_pred_abnormal_prompt.replace("{{原始流量序列}}", old_time_str).replace("{{预测流量序列}}", ori_time_str).replace("{{异常点流量}}", anomaly_points_str),
                        "input": "",
                        "output": ""
                    })
        # 设计随机窗口进行异常数据构建，总计数量与json_data保持一致
        random_samples = len(json_data)
        random_json_data = []
        random_json_data_3 = []

        for _ in tqdm(range(random_samples), desc="generate random anomaly patterns"):
            while True:
                # Randomly select a time window size between 5 and 100
                random_time_step = random.randint(5, 100)
                
                # Make sure we have enough data
                if random_time_step <= len(data):
                    # Randomly select a starting point
                    max_start = len(data) - random_time_step
                    if max_start > 0:
                        break
            start_idx = random.randint(random_time_step, max_start)
            old_tome_str = "\n".join(data[start_idx-random_time_step:start_idx])
            ori_time_str = "\n".join(data[start_idx:start_idx+random_time_step])
            
            # Find anomaly indices in the selected window
            anomaly_indices = [idx for idx in range(start_idx, start_idx+random_time_step) 
                            if idx < len(anomaly_labels) and anomaly_labels[idx] == 1]
            
            # Extract anomalous data points
            anomalous_data = [data[idx] for idx in anomaly_indices]
            
            # Determine if window contains anomalies
            has_anomalies = len(anomalous_data) > 0
            
            # Generate the appropriate output
            if has_anomalies:
                anomaly_points_str = "\n".join(anomalous_data)

                random_json_data.append({
                        "instruction": analyse_abnormal_prompt.replace("{{原始流量序列}}", ori_time_str).replace("{{异常点流量}}", anomaly_points_str),
                        "input": "",
                        "output": ""
                    })
                random_json_data_3.append({
                    "instruction": analyse_pred_abnormal_prompt.replace("{{原始流量序列}}", old_time_str).replace("{{预测流量序列}}", ori_time_str).replace("{{异常点流量}}", anomaly_points_str),
                    "input": "",
                    "output": ""
                })
            
        # Combine and shuffle the datasets
        all_json_data = json_data + random_json_data
        random.shuffle(all_json_data)

        all_json_data_3 = json_data_3 + random_json_data_3
        random.shuffle(all_json_data_3)
        # 将数据集划分为训练集和测试集
        train_data = all_json_data[:int(len(all_json_data) * 0.9)]

        test_data = all_json_data[int(len(all_json_data) * 0.9):]

        train_data_3 = all_json_data_3[:int(len(all_json_data_3) * 0.9)]
        test_data_3 = all_json_data_3[int(len(all_json_data_3) * 0.9):]

        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
        all_train_data_3.extend(train_data_3)
        all_test_data_3.extend(test_data_3)
    
    # 输出总数据个数
    print("Total number of data points:", len(all_train_data) + len(all_test_data))
    print("Total number of train_data:", len(all_train_data))
    print("Total number of test_data:", len(all_test_data))
    print("Total number of train_data_3:", len(all_train_data_3))
    print("Total number of test_data_3:", len(all_test_data_3))

    # 保存数据集
    # 保存数据集
    with open("./dataset/train/analyse_abnormal_train.json", "w", encoding="utf-8") as f:
        json.dump(all_train_data, f, ensure_ascii=False, indent=4)

    with open("./dataset/test/analyse_abnormal_test.json", "w", encoding="utf-8") as f:
        json.dump(all_test_data, f, ensure_ascii=False, indent=4)

    with open("./dataset/train/analyse_pred_abnormal_train.json.json", "w", encoding="utf-8") as f:
        json.dump(all_train_data_3, f, ensure_ascii=False, indent=4)

    with open("./dataset/test/analyse_pred_abnormal_test.json", "w", encoding="utf-8") as f:
        json.dump(all_test_data_3, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    get_forecast_dataset()
    # file_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/data/Dataset"
    # abnormal_data(file_path)






