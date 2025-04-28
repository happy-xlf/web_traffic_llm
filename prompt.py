Forecast_prompt = """以下是最近{window_size}个时间点的网络流量数据:
{ori_data}

请预测接下来{pred_size}个时间点的网络流量。
"""

Forecast_answer_prompt = """根据您提供的历史流量数据，预测接下来{pred_size}个时间点的网络流量为:
{ans_data}

"""

Abnormal_Detection_prompt = """以下是最近{window_size}个时间点的网络流量数据:
{ori_data}

请分析其中是否有异常流量数据。
"""

Abnormal_Detection_answre_yes = """通过分析您提供的数据，其中异常流量数据为:
{ans_data}

"""

Abnormal_Detection_answre_no = """通过分析您提供的数据,未见异常流量数据。
"""






