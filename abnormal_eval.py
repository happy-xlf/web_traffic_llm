import json

data = []
# with open("./abnormal_test_webLLM_ckp1400_pred.jsonl", "r", encoding="utf-8") as f:
# with open("./abnormal_test_webLLM_ckp1400_pred_tp0.jsonl", "r", encoding="utf-8") as f:
with open("./abnormal_select_test_short_webLLM_ckp1400_pred_tp0.jsonl", "r", encoding="utf-8") as f:
    for lien in f:
        data.append(json.loads(lien))
pred_list = []
ground_true = []

it = 0
for item in data:
    it += 1
    output = item["output"]
    pred = item["pred"]
    # out_nums = item["instruction"].replace("以下是最近60个时间点的网络流量数据:\n", "").replace("请分析其中是否有异常流量数据。\n", "")
    # pred_nums = pred.replace("<think>\n</think>\n通过分析您提供的数据，其中异常流量数据为:\n", "")
    # if it == 297:
    #     print(out_nums)
    #     print("==========")
    #     print(pred_nums)
    if output == "<think>\n</think>\n通过分析您提供的数据,未见异常流量数据。\n":
        ground_true.append(1)
    else:
        ground_true.append(0)
    if "通过分析您提供的数据,未见异常流量数据。" in pred:
        pred_list.append(1)
    else:
        # if out_nums == pred_nums:
        #     pred_list.append(1)
        # else:
        pred_list.append(0)

# 计算准确率，召回率，精确率和F1 score
# 计算准确率，召回率，精确率和F1 score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(ground_true, pred_list)
precision = precision_score(ground_true, pred_list)
recall = recall_score(ground_true, pred_list)
f1 = f1_score(ground_true, pred_list)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

