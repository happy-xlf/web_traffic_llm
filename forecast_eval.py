import json
import re
import numpy as np

def extract_traffic_values(text):
    # 使用正则表达式匹配流量值
    # 查找数字序列，这些数字序列位于横杠和换行符之间
    pattern = r'- (\d+)'
    
    # 查找所有匹配项
    matches = re.findall(pattern, text)
    
    # 将字符串转换为整数
    traffic_values = [int(match) for match in matches]
    
    return traffic_values

def min_max_normalize(array):
    """对数组进行最大最小归一化"""
    min_val = np.min(array)
    max_val = np.max(array)
    # 避免除以零（如果所有值都相同）
    if max_val == min_val:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

# file_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/short_ec_time_forecast_test_webLLM_tp0_ckp1400_pred.jsonl"
file_path = "/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/ec_time_forecase_test_webLLM_ckp1400_pred.jsonl"
total_mse = 0.0
total_mae = 0.0
total_raw_mse = 0.0
total_raw_mae = 0.0
sample_count = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        true_traffic = extract_traffic_values(data.get("output"))
        pred_traffic = extract_traffic_values(data.get("pred"))
        
        # Only consider common length
        pred_len = len(pred_traffic)
        true_len = len(true_traffic)
        
        if pred_len == 0:
            continue  # Skip entries with no predictions
        # Truncate true values to match prediction length
        if true_len > pred_len:
            true_traffic = true_traffic[:pred_len]
        else:
            pred_traffic = pred_traffic[:true_len]
        
        # Convert to numpy arrays for easier calculation
        true_arr = np.array(true_traffic)
        pred_arr = np.array(pred_traffic)
        
        # 保存原始MSE和MAE
        raw_mse = np.mean((true_arr - pred_arr) ** 2)
        raw_mae = np.mean(np.abs(true_arr - pred_arr))
        
        # 最大最小归一化
        # 使用两个数组的最大最小值进行归一化
        combined = np.concatenate((true_arr, pred_arr))
        min_val = np.min(combined)
        max_val = np.max(combined)
        
        if max_val == min_val:
            normalized_true = np.zeros_like(true_arr)
            normalized_pred = np.zeros_like(pred_arr)
        else:
            normalized_true = (true_arr - min_val) / (max_val - min_val)
            normalized_pred = (pred_arr - min_val) / (max_val - min_val)
        
        # Calculate MSE and MAE for normalized values
        mse = np.mean((normalized_true - normalized_pred) ** 2)
        mae = np.mean(np.abs(normalized_true - normalized_pred))
        
        # Accumulate metrics
        total_mse += mse
        total_mae += mae
        total_raw_mse += raw_mse
        total_raw_mae += raw_mae
        sample_count += 1

# Calculate average metrics
if sample_count > 0:
    avg_mse = total_mse / sample_count
    avg_mae = total_mae / sample_count
    avg_raw_mse = total_raw_mse / sample_count
    avg_raw_mae = total_raw_mae / sample_count
    
    print(f"原始数据 - Average MSE across {sample_count} samples: {avg_raw_mse:.4f}")
    print(f"原始数据 - Average MAE across {sample_count} samples: {avg_raw_mae:.4f}")
    print("\n归一化后:")
    print(f"归一化后 - Average MSE across {sample_count} samples: {avg_mse:.4f}")
    print(f"归一化后 - Average MAE across {sample_count} samples: {avg_mae:.4f}")
else:
    print("No valid samples found.")
