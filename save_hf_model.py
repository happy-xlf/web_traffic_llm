from unsloth import FastLanguageModel  # 导入FastLanguageModel类，用来加载和使用模型
import torch  # 导入torch工具，用于处理模型的数学运算
# 导入 Hugging Face Hub 的 create_repo 函数，用于创建一个新的模型仓库
from huggingface_hub import create_repo

model_name_or_path = '/mnt/tidal-nj01/dataset/xlf/web_traffic_llm/export_model/web_LLM_cpt1400'


max_seq_length = 32000  # 设置模型处理文本的最大长度，相当于给模型设置一个“最大容量”
dtype = None  # 设置数据类型，让模型自动选择最适合的精度

# 加载预训练模型，并获取tokenizer工具
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path,  # 指定要加载的模型名称
    max_seq_length=max_seq_length,  # 使用前面设置的最大长度
    dtype=dtype,  # 使用前面设置的数据类型
)
HUGGINGFACE_TOKEN = "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"

# 将模型保存为 8 位量化格式（Q8_0）
# 这种格式文件小且运行快，适合部署到资源受限的设备
if True: model.save_pretrained_gguf("WTLLM-R1-7B", tokenizer, quantization_method = "f16")

# 在 Hugging Face Hub 上创建一个新的模型仓库
create_repo("XuLiFeng/WTLLM-R1-7B", token=HUGGINGFACE_TOKEN, exist_ok=True)

# 将模型和分词器上传到 Hugging Face Hub 上的仓库
model.push_to_hub_gguf("XuLiFeng/WTLLM-R1-7B", tokenizer, token=HUGGINGFACE_TOKEN)