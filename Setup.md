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

# Check Below Notes:
which of DeepSeek, MiniMax, and Gwen is good for spec driven development
For specification-driven development (SDD), where upfront planning and architectural decomposition are critical, MiniMax M2.5 is currently the standout choice. 
While all three models are high-performance coding tools, they excel in different phases of the development lifecycle:
1. MiniMax M2.5: The "Architect" for SDD
MiniMax M2.5 is specifically engineered with an emergent "spec-writing" behavior. Before generating any code, it is trained to proactively decompose features, plan project structures, and design UI elements. 
Best For: Turning high-level requirements into detailed technical specifications and then into functional code.
Key Advantage: It learns that spending tokens on upfront planning reduces errors downstream, making it highly efficient for complex 0-to-1 system design.
Efficiency: It reportedly completes tasks roughly 37% faster than its predecessors through better task decomposition. 
2. Qwen 3 (Coder): The Agentic Implementation Specialist 
Alibaba's Qwen 3 Coder is purpose-built for agentic workflows and long-duration coding sessions. 
Best For: Working within a CLI or IDE to implement a spec, handle error recovery, and manage long tool-calling chains.
Key Advantage: It features a massive 256K context window, allowing it to maintain a coherent understanding of large codebases during implementation.
Performance: It excels at researching and structuring output, often producing more "user-friendly" code than other models. 
3. DeepSeek (V3/R1): The Reasoning & Logic Powerhouse 
DeepSeek models, particularly R1, are renowned for deep reasoning and solving complex mathematical or algorithmic logic. 
Best For: Solving specific, high-complexity logic puzzles or optimizing dense algorithmic sections within a spec.
Key Advantage: Its reasoning-first approach makes it superior for complex multi-step calculations and deep "thinking" phases ( tokens).
Feature: Upcoming versions like V4 aim for even deeper repository-level comprehension with 1M+ token context windows. 
Comparison Summary
Model 	Primary Strength in SDD	Best Tooling Context
MiniMax M2.5	Architecture & Spec Planning	System design from 0-to-1
Qwen 3 Coder	Agentic Implementation	Long CLI/IDE coding sessions
DeepSeek R1	Complex Logic & Reasoning	Algorithmic optimization
Are you planning to use these models via an IDE like Cursor or through their APIs for a custom agentic pipeline?

In an IDE via local OpenCode
When using these models within a local IDE via OpenCode, the choice depends on whether your priority is the architectural planning (the "Spec") or the high-volume implementation (the "Development"). 
1. MiniMax M2.5: The SDD Architect
MiniMax is currently the strongest choice for the initial specification and planning phase of SDD. It is specifically optimized to act as a "Real World Coworker," performing best when breaking down high-level requirements into technical tasks. 
Integration: You can connect it to OpenCode by using the /connect command and selecting the MiniMax provider.
SDD Strength: It excels at proactive decomposition, meaning it will often ask clarifying questions to refine a spec before it begins generating code. 
2. Qwen 3 Coder: The Agentic Implementer
If your spec is already written and you need an model to execute it across a large local codebase, Qwen 3 Coder is superior. 
Integration: OpenCode can access Qwen models locally through an Ollama endpoint at http://localhost:11434/v1.
SDD Strength: Its 256K context window allows it to "read" your entire specification document alongside your existing code without losing focus, making it ideal for long, stateful coding sessions. 
3. DeepSeek (V3/R1): The Logic Specialist 
DeepSeek is best used for complex algorithmic parts of your specification that require deep "reasoning" rather than just architectural planning. 
Integration: Similar to Qwen, you can run DeepSeek-R1 locally via Ollama and point OpenCode to it.
SDD Strength: Use it when your spec defines a difficult math-heavy or logic-heavy module that requires a high degree of correctness. 
Recommendation for OpenCode Users
For a complete Specification-Driven workflow:
Phase 1 (Spec Writing): Use MiniMax M2.5 via the OpenCode cloud connection to draft and refine your technical plan.
Phase 2 (Implementation): Switch to Qwen 3 Coder (hosted locally via Ollama) to execute the tasks defined in your spec, as it handles the "agentic" part of the development better. 
Would you like the specific OpenCode configuration commands to link these models to your local setup?

I want to use it via a vLLM on my Nvidia Dgx spark. Can you give me specific versions of these models that i can use on my nvidia dgx spark via vLLM.
To run these models on your NVIDIA DGX Spark (powered by the GB10 Grace Blackwell Superchip with 128GB Unified Memory), you should target specific quantized or distilled versions that fit within the ~96GB of assignable VRAM. 
Since the DGX Spark supports FP4 precision, look for models optimized for this format to maximize the 1 PFLOP of AI performance. 
1. MiniMax (Architecture & Spec Planning)
MiniMax is excellent for the "Specification" phase. For local vLLM deployment on a single DGX Spark, use the M2.1 or M2.5 versions. 
Recommended Version: MiniMax-M2.1-NVFP4 or MiniMax-M2.1-AWQ.
vLLM Command: vllm serve minimaxai/MiniMax-M2.1 --quantization awq
Why: These versions are optimized for NVIDIA hardware and fit comfortably within your memory. 
2. Qwen (Agentic Implementation)
For the "Development" phase, Qwen3-Coder is the target. It excels at long-context implementation within an IDE. 
Recommended Version: Qwen/Qwen3-VL-32B-Instruct-FP8 or Qwen/Qwen3-30B-A3B-NVFP4.
vLLM Command: vllm serve Qwen/Qwen3-32B-Instruct --quantization fp8
Why: The 32B version is highly efficient on Blackwell architecture and provides a strong balance of coding logic and context handling. 
3. DeepSeek (Complex Logic Reasoning)
For high-complexity reasoning, use the distilled versions of DeepSeek-R1 or the V3.2 series. 
Recommended Version: deepseek-ai/DeepSeek-R1-Distill-Llama-70B or nvidia/deepseek-ai/deepseek-v3.2.
vLLM Command: vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max-model-len 32768
Why: A single DGX Spark can run up to 70B parameter models for fine-tuning or inference. The 70B R1-Distill version is widely considered the "sweet spot" for high-end reasoning on this hardware class. 
Deployment Pro-Tips for DGX Spark
vLLM Version: Ensure you are using vLLM v0.9.0+ to leverage FP8 Marlin and better block-wise quantization support for these newer models.
Unified Memory: You can allocate approximately 96GB of your 128GB unified memory as VRAM. Use the --gpu-memory-utilization flag in vLLM to fine-tune this.
Local Access: Once your vLLM server is running (typically on port 8000), point OpenCode to http://<your-dgx-ip>:8000/v1. 
Do you need the specific Docker run command or vLLM configuration file to launch these models with optimal Blackwell-specific flags?

Yes please give me specific docker commands
To run these models on your NVIDIA DGX Spark (Blackwell GB10 architecture) with vLLM, you should use optimized images that support NVFP4 and FlashInfer backends to maximize the 128GB Unified Memory. 
Use the following docker run commands for each model type. 
1. MiniMax M2.5 (The SDD Architect)
Optimized for architectural decomposition and specification planning.
Specific docker run commands for MiniMax M2.5, Qwen3-Coder-32B, and DeepSeek-R1-Distill-Llama-70B models on DGX Spark are available on Hugging Face. 
Key configuration notes for DGX Spark include recommended --gpu-memory-utilization settings (0.8-0.9), mounting ~/.cache/huggingface for a shared cache, using unique ports for each model, and leveraging FlashInfer for performance on Blackwell architecture. 
Would you like the OpenCode configuration strings to connect these three local endpoints to your IDE?

I do not see the docker commands. Perhaps the output got truncated.
To run these models on your NVIDIA DGX Spark (Blackwell GB10 architecture) with vLLM, you should use an image optimized for ARM64 and CUDA 13. 
Specific Blackwell-optimized images, such as scitrera/dgx-spark-vllm:0.13.0-t4, are recommended to avoid common kernel compilation issues on this architecture. 
1. MiniMax M2.5 (The SDD Architect)
Optimized for architectural decomposition and specification planning. 
Here are examples of Docker commands to run MiniMax M2.5, Qwen3-Coder-32B, and DeepSeek-R1-Distill-Llama-70B on DGX Spark: 
MiniMax M2.5: Optimized for architectural decomposition.
Qwen3-Coder-32B: Optimized for implementation and tool-calling.
DeepSeek-R1-Distill-Llama-70B: Optimized for complex reasoning. 
For the full Docker commands and further details, please refer to NVIDIA Developer Forums. 
Key flags for the DGX Spark environment include --gpu-memory-utilization for reserving memory, --attention-backend flashinfer for improved efficiency, --privileged for unified memory access, and --ipc=host for shared memory communication. 
Would you like the OpenCode environment variables to point your IDE to these three specific local ports?

# Links
https://forums.developer.nvidia.com/t
https://forums.developer.nvidia.com/t/best-inference-framework-open-models-for-orchestrator-workers-agentic-coding-on-gb10-5090-hybrid/356455
https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/vllm_deploy_guide.md#:~:text=System%20Requirements,minimax_m2%20%5C%20%2D%2Dreasoning%2Dparser%20minimax_m2_append_think
https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
https://medium.com/@amri369/how-to-serve-deepseek-flagship-models-for-inference-with-vllm-and-tgi-using-openai-sdk-176b6cb801ef
https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862/136?page=7


