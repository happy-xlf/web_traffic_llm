
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /mnt/tidal-nj01/dataset/xlf/web_traffic_llm/export_model/web_LLM_cpt1400 \
    --dtype auto \
    --port 8666 \
    --served-model-name WTLLM-R1-7B \
    --tensor-parallel-size 4 \
    --max_model_len 20000 \
    --gpu_memory_utilization 0.8

    