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

# Cleanup stopped vLLM's and remove the image from local docker
docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm 

# Use one of the following AI Models and run them on different ports:
meta-llama/Llama-3.1-8B-Instruct
docker run -it --gpus all -p 8000:8000 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/gb10_ownera/models/hub:/root/.cache/huggingface/hub nvcr.io/nvidia/vllm:26.01-py3 vllm serve meta-llama/Llama-3.1-8B-Instruct

mistralai/Mistral-7B-Instruct-v0.3
docker run -it --gpus all -p 8001:8000 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/gb10_ownera/models/hub:/root/.cache/huggingface/hub nvcr.io/nvidia/vllm:26.01-py3 vllm serve mistralai/Mistral-7B-Instruct-v0.3

mistralai/Mixtral-8x7B-Instruct-v0.1
docker run -it --gpus all -p 8002:8000 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/gb10_ownera/models/hub:/root/.cache/huggingface/hub nvcr.io/nvidia/vllm:26.01-py3 vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1


# Top Coding Models for DGX Spark:
1. NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4
Qwen3-Coder-30B-A3B (or 32B): Highly regarded for performance-to-size ratio on the Spark, especially when using NVFP4 precision for high throughput.
docker run -it --gpus all -e HF_TOKEN=$HF_TOKEN \
-p 8003:8000 \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v /home/gb10_ownera/models/hub:/root/.cache/huggingface/hub \
nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
vllm serve NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4 --enable-auto-tool-choice  --tool-call-parser=qwen3_coder

2. GadflyII/GLM-4.6V-NVFP4
GLM-4.6V (106B or similar): A 106B parameter MoE model that is recommended for high-quality coding tasks, offering decent speeds on the Spark's architecture.

3. nvidia/DeepSeek-V3.2-NVFP4
DeepSeek Coder V2: A strong, popular choice for code generation, noted for its efficiency, even in larger parameter sizes.

# Best GLM Models for DGX Spark:
1. zai-org/GLM-4.7-Flash : GLM-4.7 Flash: Recognized as a top performer for local coding and RAG tasks on DGX Spark, providing ~2000 t/s (prefill) and ~55 t/s (generation).

2. zai-org/GLM-4.6V-Flash : GLM-4.6V (106B, 12B Active): A MoE model that runs well, particularly in 4-bit AWQ or FP8 on single/dual sparks.

3. Llama 3.1 8B: Recommended for maximum speed and lower-latency tasks.

4. GPT-OSS 120B: Suitable for complex tasks requiring high capability, though it is better for prototyping than high-throughput production.

Check Below Notes:
1. Qwen 2.5 / Qwen 3 Coder (Best Overall for Coding) 
Why it's best for Specs: Qwen models excel at interpreting complex, nuanced requirements and turning them into functional code, maintaining a high balance between creativity and practical application.
Key Strengths: It consistently produces better-structured, user-friendly, and maintainable code. It is highly reliable in tool-calling scenarios.
Context: With up to 256K-1M context windows, it is excellent for repository-level analysis, allowing it to understand how a specific feature fits into a larger codebase. 
2. DeepSeek (Best for Algorithmic Logic & Efficiency) 
Why it's good for Specs: If your specifications require complex algorithms, mathematical computations, or tight, logical reasoning, DeepSeek (specifically V3 or R1) is highly effective.
Key Strengths: DeepSeek-Coder-V2 excels in competitive programming benchmarks and provides high-speed, cost-efficient inference.
Verdict: Ideal for generating the backend logic or data structures derived from a spec, though it might require more prompting for UI polish compared to Qwen. 
3. MiniMax (Best for Agentic Workflows) 
Why it's good for Specs: According to recent 2026 data, MiniMax M2.5 has shown exceptional performance in agentic benchmarks (SWE-Bench Verified), making it highly effective at working autonomously as a "coworker".
Key Strengths: It is highly efficient and excels at working within IDE agents (like Cline) to handle multi-file operations. 
Summary Recommendation
For Feature Development/Full-Stack Specs: Qwen 3 Coder (highest code quality and structure).
For Algorithmic/Mathematical Specs: DeepSeek-V3 (best reasoning and speed).
For Agentic/Self-Driving Development: MiniMax M2.5 (best at multi-file debugging). 

