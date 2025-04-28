import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import numpy as np
import jieba
import re
import json

def calculate_bleu(references, predictions, smooth=True):
    """
    计算BLEU分数（使用jieba中文分词）
    
    参数:
    - references: 参考文本列表，每项是一个参考文本(或多参考情况下是参考文本列表)
    - predictions: 预测文本列表
    - smooth: 是否使用平滑函数
    
    返回:
    - 包含BLEU-1到BLEU-4分数的字典
    """
    if smooth:
        smoothing = SmoothingFunction().method1
    else:
        smoothing = None
    
    # 处理输入格式，使用jieba分词
    processed_refs = []
    for ref in references:
        if isinstance(ref[0], list):  # 已经是分词后的列表
            processed_refs.append(ref)
        else:  # 需要分词
            processed_refs.append([list(jieba.cut(r)) for r in ref])
    
    processed_preds = []
    for pred in predictions:
        if isinstance(pred, list):  # 已经是分词后的列表
            processed_preds.append(pred)
        else:  # 需要分词
            processed_preds.append(list(jieba.cut(pred)))
    
    # 计算不同n-gram的BLEU分数
    bleu_scores = {}
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    for n, weights in enumerate(weights_list, start=1):
        score = corpus_bleu(processed_refs, processed_preds, weights=weights, smoothing_function=smoothing)
        bleu_scores[f'BLEU-{n}'] = score
    
    return bleu_scores

def calculate_rouge(references, predictions):
    """
    计算ROUGE分数（使用jieba分词）
    
    参数:
    - references: 参考文本列表
    - predictions: 预测文本列表
    
    返回:
    - 包含ROUGE-1, ROUGE-2和ROUGE-L分数的字典
    """
    rouge = Rouge()
    
    # 使用jieba进行分词，并将分词结果用空格连接
    refs = []
    for ref in references:
        if isinstance(ref[0], list):
            refs.append(' '.join(ref[0]))
        else:
            # 使用jieba分词，并保留标点符号
            segmented = ' '.join(list(jieba.cut(ref[0])))
            refs.append(segmented)
    
    preds = []
    for pred in predictions:
        if isinstance(pred, list):
            preds.append(' '.join(pred))
        else:
            # 使用jieba分词，并保留标点符号
            segmented = ' '.join(list(jieba.cut(pred)))
            preds.append(segmented)
    
    # 计算每对文本的ROUGE分数并取平均值
    scores_all = []
    for ref, pred in zip(refs, preds):
        try:
            # 确保文本非空
            if not ref or not pred:
                print("警告: 空文本被跳过")
                continue
                
            scores = rouge.get_scores(pred, ref)[0]
            scores_all.append(scores)
        except Exception as e:
            print(f"计算ROUGE时出错: {e}")
            print(f"参考文本: {ref}")
            print(f"预测文本: {pred}")
    
    # 计算平均分
    rouge_scores = {}
    if scores_all:
        rouge_1 = np.mean([score['rouge-1']['f'] for score in scores_all])
        rouge_2 = np.mean([score['rouge-2']['f'] for score in scores_all])
        rouge_l = np.mean([score['rouge-l']['f'] for score in scores_all])
        
        rouge_scores = {
            'ROUGE-1': rouge_1,
            'ROUGE-2': rouge_2,
            'ROUGE-L': rouge_l
        }
    
    return rouge_scores

def evaluate_text_quality(references, predictions, use_char_level=False):
    """
    综合评估文本质量：计算BLEU和ROUGE分数
    
    参数:
    - references: 参考文本列表，每项可以是单个字符串或字符串列表(多参考)
    - predictions: 预测文本列表，每项是一个字符串
    - use_char_level: 是否使用字符级别分词(适用于某些中文任务)
    
    返回:
    - 包含所有评估指标的字典
    """
    # 准备参考文本格式，适应多参考情况
    formatted_refs = []
    for ref in references:
        if isinstance(ref, str):
            formatted_refs.append([ref])
        else:
            formatted_refs.append(ref)
    
    # 如果使用字符级别分词
    if use_char_level:
        char_refs = []
        for refs in formatted_refs:
            char_refs.append([list(ref) for ref in refs])
        
        char_preds = [list(pred) for pred in predictions]
        
        # 计算BLEU分数(字符级)
        bleu_scores = calculate_bleu(char_refs, char_preds)
    else:
        # 计算BLEU分数(词级，使用jieba分词)
        bleu_scores = calculate_bleu(formatted_refs, predictions)
    
    # 计算ROUGE分数
    rouge_scores = calculate_rouge(formatted_refs, predictions)
    
    # 合并结果
    results = {}
    results.update(bleu_scores)
    results.update(rouge_scores)
    
    return results

def batch_evaluate(references_file, predictions_file):
    """
    批量评估两个文件中的文本
    
    参数:
    - references_file: 参考文本文件路径，每行一个文本
    - predictions_file: 预测文本文件路径，每行一个文本
    
    返回:
    - 包含所有评估指标的字典
    """
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f if line.strip()]
    
    # 转换格式为列表的列表
    references = [[ref] for ref in references]
    
    # 确保两个列表长度一致
    min_len = min(len(references), len(predictions))
    references = references[:min_len]
    predictions = predictions[:min_len]
    
    print(f"评估 {min_len} 对文本...")
    
    # 评估
    results_word = evaluate_text_quality(references, predictions, use_char_level=False)
    results_char = evaluate_text_quality(references, predictions, use_char_level=True)
    
    return results_word, results_char

# 示例使用
if __name__ == "__main__":
    # 示例数据
    # pred_file = "./web_traffic_select_test_pred.jsonl"
    # pred_file = "./web_traffic_select_test_WebLLM_ckp1400_pred.jsonl"
    pred_file = "./analyse_test_webLLM_ckp1400_pred.jsonl"
    references = []
    predictions = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            references.append(json.loads(line)["output"])
            predictions.append(json.loads(line)["pred"])
    
    print("使用词级jieba分词评估:")
    results_word = evaluate_text_quality(references, predictions, use_char_level=False)
    for metric, score in results_word.items():
        print(f"{metric}: {score:.4f}")
    
    # print("\n使用字符级分词评估:")
    # results_char = evaluate_text_quality(references, predictions, use_char_level=True)
    # for metric, score in results_char.items():
    #     print(f"{metric}: {score:.4f}")

