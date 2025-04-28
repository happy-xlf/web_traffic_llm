import requests
import os
from openai import OpenAI
import json
from tqdm import tqdm
import re
import concurrent.futures
import threading
import pandas as pd
from prompt import Forecast_prompt, Forecast_answer_prompt,Abnormal_Detection_prompt, Abnormal_Detection_answre_yes, Abnormal_Detection_answre_no
from tqdm import tqdm
import random
import numpy as np

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

def gen_forecast_time(data, train_path, test_path):
    time_steps = [10,15,20,25,30]
    forecast_steps = [1,3,5,7,10]
    json_data = []
    # 基于既定的规则获取
    for time_step in tqdm(time_steps, total=len(time_steps), desc="generate"):
        for forecast_step in forecast_steps:
            for i in range(0,len(data) - time_step - forecast_step + 1, time_step//2):
                ori_time_str = "\n".join(data[i:i+time_step])
                ans_data_str = "\n".join(data[i+time_step:i+time_step+forecast_step])
                json_data.append({
                    "instruction": Forecast_prompt.replace("{window_size}", str(time_step)).replace("{ori_data}", ori_time_str).replace("{pred_size}", str(forecast_step)),
                    "input": "",
                    "output": "<think>\n</think>\n" + Forecast_answer_prompt.replace("{pred_size}", str(forecast_step)).replace("{ans_data}", ans_data_str)
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
                
                random_json_data.append({
                    "instruction": Forecast_prompt.replace("{window_size}", str(random_time_step)).replace("{ori_data}", ori_time_str).replace("{pred_size}", str(random_forecast_step)),
                    "input": "",
                    "output": "<think>\n</think>\n" + Forecast_answer_prompt.replace("{pred_size}", str(random_forecast_step)).replace("{ans_data}", ans_data_str)
                })
    # random dataset and split train_dataset and test_dataset

    # 1.Shuffle the dataset randomly
    random.shuffle(json_data)
    random.shuffle(random_json_data)
    
    # 2.Split into training and test sets (80% train, 20% test)
    split_idx = int(len(json_data) * 0.9)
    train_data = json_data[:split_idx]
    test_data = json_data[split_idx:][:250]

    random_split_idx = int(len(random_json_data) * 0.9)
    random_train_data = random_json_data[:random_split_idx]
    random_test_data = random_json_data[random_split_idx:][:250]

    train_data = train_data + random_train_data
    test_data = test_data + random_test_data
    
    # 3.Save the splits
    # with open(train_path, "w", encoding="utf-8") as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)
    
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    print(f"Total data: {len(json_data)}")
    print(f"Training data: {len(train_data)} saved to {train_path}")
    print(f"Test data: {len(test_data)} saved to {test_path}")



def get_forecast_dataset():
    file_path = "./data/ec_data.csv"
    train_path = "./dataset/train/ec_time_forecast_train.json"
    test_path = "./dataset/test/short_ec_time_forecast_test.json"
    data = get_data(file_path)
    gen_forecast_time(data, train_path, test_path)

    file_path = "./data/uk_data.csv"
    train_path = "./dataset/train/uk_time_forecast_train.json"
    test_path = "./dataset/test/short_uk_time_forecast_test.json"
    data = get_data(file_path)
    gen_forecast_time(data, train_path, test_path)

def get_detection_dataset(file_path, train_path, test_path):
   
    # 获取文件夹下的所有文件名
    all_files = os.listdir(file_path)
    
    # Filter for CSV files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    all_train_data = []
    all_test_data = []
    
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
        # 基于既定的规则获取
        for time_step in tqdm(time_steps, total=len(time_steps), desc="generate"):
            for i in range(0, len(data) - time_step + 1, time_step):
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
                    output_content = Abnormal_Detection_answre_yes.replace("{ans_data}", anomaly_points_str)
                else:
                    output_content = Abnormal_Detection_answre_no
                
                json_data.append({
                    "instruction": Abnormal_Detection_prompt.replace("{window_size}", str(time_step)).replace("{ori_data}", ori_time_str),
                    "input": "",
                    "output": "<think>\n</think>\n" + output_content
                })
        # 设计随机窗口进行异常数据构建，总计数量与json_data保持一致
        random_samples = len(json_data)
        random_json_data = []

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
            start_idx = random.randint(0, max_start)
            
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
                output_content = Abnormal_Detection_answre_yes.replace("{ans_data}", anomaly_points_str)
            else:
                output_content = Abnormal_Detection_answre_no
            
            random_json_data.append({
                "instruction": Abnormal_Detection_prompt.replace("{window_size}", str(random_time_step)).replace("{ori_data}", ori_time_str),
                "input": "",
                "output": "<think>\n</think>\n" + output_content
            })
            
        # Combine and shuffle the datasets
        all_json_data = json_data + random_json_data
        random.shuffle(all_json_data)
        # Split into training and test sets (90% train, 10% test)
        
        index_label_data = []
        others_data = []
        for idx,it in enumerate(all_json_data):
            if "其中异常流量数据为:" in it["output"]:
                index_label_data.append(it)
            else:
                others_data.append(it)
        if len(index_label_data) == 0:
            all_json_data = all_json_data[:len(all_json_data)//5]
            split_idx = int(len(all_json_data) * 0.9)
            train_data = all_json_data[:split_idx]
            test_data = all_json_data[split_idx:]
        else:
            others_data = others_data[:len(index_label_data)]
            others_data.extend(index_label_data)
            random.shuffle(others_data)
            split_idx = int(len(others_data) * 0.9)
            train_data = others_data[:split_idx]
            test_data = others_data[split_idx:]

        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
        
        print(f"当前文件: {csv_file}")
        print(f"Total data: {len(all_json_data)}")
        print(f"Training data: {len(train_data)}")
        print(f"Test data: {len(test_data)}")
        # except Exception as e:
        #     print(f"Error loading {csv_file}: {e}")

   
    # 3.Save the splits
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(all_train_data, f, ensure_ascii=False, indent=4)
    
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(all_test_data, f, ensure_ascii=False, indent=4)
    
    print(f"Total data: {len(all_train_data) + len(all_test_data)}")
    print(f"Training data: {len(all_train_data)} saved to {train_path}")
    print(f"Test data: {len(all_test_data)} saved to {test_path}")
    

def main():
    # file_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/data/Dataset"
    # train_path = "./dataset/train/abnormal_train.json"
    # test_path = "./dataset/test/abnormal_test.json"
    # get_detection_dataset(file_path, train_path, test_path)
    get_forecast_dataset()


if __name__ == "__main__":
    main()






