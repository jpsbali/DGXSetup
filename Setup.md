# Setup vLLM
Configure your HF_TOKEN and export it
export HF_TOKEN=XYZ
export LATEST_VLLM_VERSION=26.01-py3
docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}

# Run vLLM
docker run -it --gpus all -e HF_TOKEN=$HF_TOKEN \
-p 8000:8000 \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v /home/gb10_ownera/models/hub:/root/.cache/huggingface/hub \
nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
vllm serve meta-llama/Llama-3.1-8B-Instruct

# Cleanup topped vLLM's and remove image from docker
docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm 

# Use one of the followoing AI Models and run them on different ports:
meta-llama/Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
mistralai/Mixtral-8x7B-Instruct-v0.1

# Best GLM Models for DGX Spark:

zai-org/GLM-4.7-Flash : GLM-4.7 Flash: Recognized as a top performer for local coding and RAG tasks on DGX Spark, providing ~2000 t/s (prefill) and ~55 t/s (generation).

zai-org/GLM-4.6V-Flash : GLM-4.6V (106B, 12B Active): A MoE model that runs well, particularly in 4-bit AWQ or FP8 on single/dual sparks.

NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4


Top Coding Models for DGX Spark:
NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4
Qwen3-Coder-30B-A3B (or 32B): Highly regarded for performance-to-size ratio on the Spark, especially when using NVFP4 precision for high throughput.

GadflyII/GLM-4.6V-NVFP4
GLM-4.6V (106B or similar): A 106B parameter MoE model that is recommended for high-quality coding tasks, offering decent speeds on the Spark's architecture.

nvidia/DeepSeek-V3.2-NVFP4
DeepSeek Coder V2: A strong, popular choice for code generation, noted for its efficiency, even in larger parameter sizes.

nvidia/DeepSeek-V3.2-NVFP4
Llama 3.1 8B: Recommended for maximum speed and lower-latency tasks.
GPT-OSS 120B: Suitable for complex tasks requiring high capability, though it is better for prototyping than high-throughput production. 

Install OpenCode 
1. curl -fsSL https://opencode.ai/install | bash
2. Configure "opencode" extention in VS Code from 
