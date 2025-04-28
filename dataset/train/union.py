import json
import random


# with open("abnormal_train.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# with open("ec_time_forecast_train.json", "r", encoding="utf-8") as f:
#     ec_time_forecast_train = json.load(f)

# with open("uk_time_forecast_train.json", "r", encoding="utf-8") as f:
#     uk_time_forecast_train = json.load(f)

# analyse_data = []
# with open("analyse_train.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         analyse_data.append(json.loads(line))

# analyse_data = analyse_data + data + ec_time_forecast_train + uk_time_forecast_train
# random.shuffle(analyse_data)

with open("web_traffic_train.json", "r", encoding="utf-8") as f:
    # json.dump(analyse_data, f, ensure_ascii=False)
    analyse_data = json.load(f)



lens = []
for it in analyse_data:
    lens.append(len(it["instruction"]))
print(lens)
with open("lens.txt", "w", encoding="utf-8") as f:
    f.write(str(lens))
# 统计最大最小值，以及分布情况
print("最大长度：", max(lens))
print("最小长度：", min(lens))
print("平均长度：", sum(lens) / len(lens))
print("长度分布：", len(set(lens)))
# matplotlib直方图可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(lens, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Instruction Lengths')
plt.xlabel('Length (characters)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

# Save the figure to a file
plt.savefig('../test/instruction_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

print("Histogram saved as 'instruction_length_distribution.png'")

