import json


with open("./web_traffic_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

select_data = []
for it in data:
    instr = it["instruction"]
    if "以下是最近" in instr:
        num_traffic = instr.split("以下是最近")[-1].split("个时间点的网络流量数据:\n")[0]
        num = int(num_traffic)
        if "请预测接下来" in instr:
            pred_lens = int(instr.split("请预测接下来")[-1].split("个时间点的网络流量。\n")[0])
            if pred_lens<=30:
                select_data.append(it)
        else:
            if num <= 30:
                select_data.append(it)
    else:
        if len(instr) < 1000:
            select_data.append(it)

with open("web_traffic_select_test.json", "w", encoding="utf-8") as f:
    json.dump(select_data, f, ensure_ascii=False)


