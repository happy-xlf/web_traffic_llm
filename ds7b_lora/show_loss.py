import matplotlib.pyplot as plt
import json
import matplotlib as mpl

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'  # 数学公式字体兼容性设置

# 读取和解析数据
steps = []
losses = []

# 直接从输入中解析JSON数据
with open("7b_forecast_abnormal_epoch5/trainer_log1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        steps.append(data['current_steps'])
        losses.append(data['loss'])

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(steps, losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.grid(True)
# plt.tight_layout()

# 保存图像
plt.savefig('loss_curve.png', dpi=300)

# 显示图像
plt.show()

print(f"损失从 {losses[0]} 下降到 {losses[-1]}，总训练步数: {steps[-1]}")
