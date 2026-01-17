model_pricing = {
    # --- OpenAI LLMs ---
    "gpt-4-0125-preview": {"input_price": 10.00, "output_price": 30.00},
    "gpt-4-turbo-preview": {"input_price": 10.00, "output_price": 30.00},
    "gpt-4-1106-preview": {"input_price": 10.00, "output_price": 30.00},
    "gpt-4-1106-vision-preview": {"input_price": 10.00, "output_price": 30.00},
    "gpt-4": {"input_price": 30.00, "output_price": 60.00},
    "gpt-4-32k": {"input_price": 60.00, "output_price": 120.00},
    "gpt-4o-mini": {"input_price": 0.15, "output_price": 0.60},
    "gpt-4o": {"input_price": 2.50, "output_price": 10.00},
    "gpt-4o-2024-08-06": {"input_price": 2.50, "output_price": 10.00},
    "gpt-4o-mini-2024-07-18": {"input_price": 0.15, "output_price": 0.60},
    "gpt-4.1-mini-2025-04-14": {"input_price": 0.40, "output_price": 1.60},
    "gpt-4.1-mini": {"input_price": 0.40, "output_price": 1.60},
    "gpt-5-mini": {"input_price": 0.25, "output_price": 2.00},
    "gpt-5": {"input_price": 1.25, "output_price": 10.00},
    "gpt-5-mini-2025-08-07": {"input_price": 0.25, "output_price": 2.00},
    "o3-mini-2025-01-31": {"input_price": 1.10, "output_price": 4.40},
    "ft:gpt-4o-mini-2024-07-18": {"input_price": 0.30, "output_price": 1.20},
    "gpt-3.5-turbo-0125": {"input_price": 0.50, "output_price": 1.50},
    "gpt-3.5-turbo": {"input_price": 0.50, "output_price": 1.50},
    "gpt-3.5-turbo-instruct": {"input_price": 1.50, "output_price": 2.00},
    "gpt-3.5-turbo-1106": {"input_price": 1.00, "output_price": 2.00},
    "gpt-3.5-turbo-0613": {"input_price": 1.50, "output_price": 2.00},
    "gpt-3.5-turbo-16k-0613": {"input_price": 3.00, "output_price": 4.00},
    "gpt-3.5-turbo-0301": {"input_price": 1.50, "output_price": 2.00},
    # --- OpenAI Embeddings ---
    # Output price is 0 because embeddings return vectors, not tokens.
    "text-embedding-3-small": {"input_price": 0.02, "output_price": 0.00},
    "text-embedding-3-large": {"input_price": 0.13, "output_price": 0.00},
    "text-embedding-ada-002": {"input_price": 0.10, "output_price": 0.00},
    # --- Anthropic ---
    "claude-3-haiku-20240307-v1": {"input_price": 0.25, "output_price": 1.25},
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input_price": 0.25,
        "output_price": 1.25,
    },
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_price": 1.00,
        "output_price": 5.00,
    },
    "claude-3-5-sonnet-20240620-v1": {"input_price": 3.00, "output_price": 15.00},
    "claude-3-7-sonnet-20250219": {"input_price": 3.00, "output_price": 15.00},
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_price": 3.00,
        "output_price": 15.00,
    },
    # --- Google ---
    "gemini-1.5-flash-preview-0514": {"input_price": 0.50, "output_price": 1.50},
    # Google Embeddings (Estimated Vertex AI pricing; often free in AI Studio)
    "text-embedding-004": {"input_price": 0.10, "output_price": 0.00},
    # --- Meta (Generic/Direct) ---
    "meta.llama3-8b-instruct-v1": {"input_price": 0.40, "output_price": 0.60},
    "meta.llama3-1-70b-instruct-v1:0": {"input_price": 2.65, "output_price": 3.50},
    "meta.llama3-1-8b-instruct-v1:0": {"input_price": 0.30, "output_price": 0.60},
    # =========================================================================
    # --- NEBIUS AI MODELS ---
    # Pricing is for "Base" flavor (per 1M tokens).
    # Model IDs use the format typically required by the Nebius/OpenAI-compatible API.
    # =========================================================================
    # --- Nebius LLMs (DeepSeek) ---
    "deepseek-ai/DeepSeek-V3": {"input_price": 0.50, "output_price": 1.50},
    "deepseek-ai/DeepSeek-R1": {"input_price": 0.80, "output_price": 2.40},
    # --- Nebius LLMs (Meta Llama) ---
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "input_price": 1.00,
        "output_price": 3.00,
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "input_price": 0.13,
        "output_price": 0.40,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "input_price": 0.02,
        "output_price": 0.06,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {"input_price": 0.13, "output_price": 0.40},
    "meta-llama/Llama-3.2-90B-Vision-Instruct": {
        "input_price": 0.13,
        "output_price": 0.40,
    },
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {
        "input_price": 0.03,
        "output_price": 0.09,
    },
    # --- Nebius LLMs (Qwen) ---
    "Qwen/Qwen2.5-72B-Instruct": {"input_price": 0.13, "output_price": 0.40},
    "Qwen/Qwen2.5-32B-Instruct": {"input_price": 0.06, "output_price": 0.20},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"input_price": 0.06, "output_price": 0.20},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"input_price": 0.03, "output_price": 0.09},
    "Qwen/Qwen-2-VL-72B-Instruct": {"input_price": 0.13, "output_price": 0.40},
    # --- Nebius LLMs (Other) ---
    "mistralai/Mistral-Nemo-Instruct-2407": {"input_price": 0.08, "output_price": 0.24},
    "google/gemma-2-9b-it": {"input_price": 0.03, "output_price": 0.09},
    "google/gemma-2-27b-it": {"input_price": 0.27, "output_price": 0.27},
    "microsoft/Phi-3.5-mini-instruct": {"input_price": 0.02, "output_price": 0.06},
    # --- Nebius Embeddings ---
    # Nebius hosts Qwen3-Embedding at an extremely low rate.
    "Qwen/Qwen3-Embedding-8B": {"input_price": 0.01, "output_price": 0.00},
    # Other embedding models available on Nebius
    # (e.g., BAAI/bge-en-icl, intfloat/e5-mistral).
    # Precise pricing for BGE/E5 on Nebius varies by flavor
    # but is generally competitive (~$0.01-0.02).
    # We include Qwen3 above as the primary priced embedding model.
}
