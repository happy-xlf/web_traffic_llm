import json

file_path = "./web_traffic_train.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 在这里处理数据
print(len(data))
