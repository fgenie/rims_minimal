# vllm serving script used for the experiments
ext_ip=7870
docker run --runtime nvidia --gpus '"device=0,1,2,3"' --rm  --name localserve_vllm \
    -v /home/taeng/data:/root/.cache/huggingface \
    -p $ext_ip:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    --ipc=host \
    vllm/vllm-openai:v0.4.2 \
    --trust-remote-code \
    --model meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 8100 \
    --gpu-memory-utilization 0.5 --tensor-parallel 4 --disable-custom-all-reduce # https://github.com/vllm-project/vllm/issues/4430#issuecomment-2127639727
