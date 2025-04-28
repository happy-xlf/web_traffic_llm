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
from datetime import datetime
from domain_prompt import base_prompt, analyse_abnormal_prompt, analyse_pred_abnormal_prompt
import logging

# Set up logging
log_filename = f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key="xxx",
    base_url="http://0.0.0.0:8666/v1",
)

def get_ans(idx, data):
    try:
        content = data["instruction"]
        completion = client.chat.completions.create(
            # model="DeepSeek-R1-32B",
            model = "WTLLM-R1-7B",
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0.8,  # You can adjust this value between 0 and 2
            top_p=0.9        # You can adjust this value between 0 and 1
        )
        predict_content = completion.choices[0].message.content
        # data["output"] = "<think>\n" + predict_content
        data["pred"] = predict_content
        return idx, data
    except Exception as e:
        logger.error(f"Error in get_ans: {str(e)}")
        # data["output"] = "error"
        data["pred"] = "error"
        return idx, data

# 线程安全的写入锁
file_lock = threading.Lock()

def write_result(result, output_file):
    """线程安全地将结果写入文件"""
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def get_think_ans(folder_path, output_file):
    jsdata = []
    # for file in tqdm(os.listdir(folder_path)):
    #     path = os.path.join(folder_path, file)
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     jsdata.extend(data)
    if folder_path.split(".")[-1] == "json":
        with open(folder_path, "r", encoding="utf-8") as f:
            jsdata = json.load(f)
    else:
        with open(folder_path, "r", encoding="utf-8") as f:
            for line in f:
                jsdata.append(json.loads(line))
    
    # 设置线程池的最大线程数
    max_workers = 20  # 根据系统资源和API限制调整此值
    
    # 创建进度显示
    progress_bar = tqdm(total=len(jsdata), desc="Processing")
    
    # 已完成计数器 (线程安全)
    completed = 0
    completed_lock = threading.Lock()
    
    def update_progress(_):
        nonlocal completed
        with completed_lock:
            completed += 1
            progress_bar.update(1)
    
    # 使用线程池执行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 存储所有future对象
        future_to_idx = {executor.submit(get_ans, idx, data): idx for idx, data in enumerate(jsdata)}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                idx, result = future.result()
                # if "error" not in result["output"]:
                if "error" not in result["pred"]:
                    write_result(result, output_file)
                else:
                    logger.error(f"Error processing item {idx}: {result['pred']}")
                    # 可选：将错误记录到单独的日志文件
            except Exception as exc:
                logger.error(f"Item {idx} generated an exception: {exc}")
            finally:
                update_progress(1)
    
    progress_bar.close()
    print(f"处理完成，结果已保存到 {output_file}")

def main():
    # folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/train/analyse"
    # save_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/train/analyse_train.jsonl"
    # folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/analyse"
    # save_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/analyse_test.jsonl"
    # folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/web_traffic_select_test.json"
    # save_path = "./web_traffic_select_test_WebLLM_ckp1400_pred.jsonl"
    # folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/abnormal_test.json"
    # save_path = "./abnormal_test_pred.jsonl"


    # folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/short_ec_time_forecast_test.json"
    # save_path = "./short_ec_time_forecast_test_webLLM_tp0_ckp1400_pred.jsonl"
    # folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/abnormal_select_test_short.json"
    # save_path = "./abnormal_select_test_short_webLLM_ckp1400_pred_tp0_3.jsonl"
    folder_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/dataset/test/analyse_test.jsonl"
    save_path = "./analyse_test_webLLM_ckp1400_pred_tp0_8.jsonl"
    get_think_ans(folder_path, save_path)


if __name__ == "__main__":
    main()
    