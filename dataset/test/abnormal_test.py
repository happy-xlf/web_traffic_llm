import json


with open("./abnormal_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for it in data:
    if "0.0" in it["instruction"] or len(it["output"])>60 or len(it["instruction"]) > 1000: 
        continue
    new_data.append(it)

with open("./abnormal_select_test_short.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

