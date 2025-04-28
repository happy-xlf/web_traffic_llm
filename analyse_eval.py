import json

compare_prompt = """你是一位资深的评审专家，你需要对两个模型的回答结果进行评判，选出最好的模型，并给出理由。
# 输入内容
以下是问题和两个模型的回答结果：
问题：
{question}
模型A的回答：
{model_a_answer}
模型B的回答：
{model_b_answer}

# 输出格式
{
    "reason": "评判的理由",
    "best_model": "模型A" 或 "模型B"
}
请根据回答的质量、准确性、逻辑性、流畅性等方面进行评判，给出理由和最好的模型。
"""
