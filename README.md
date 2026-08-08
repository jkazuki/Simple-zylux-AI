# Custom Transformer Dual-Model AI

This repository contains a complete, from-scratch implementation of a Transformer-based Large Language Model (LLM) ecosystem. It features a custom Dynamic Byte-Pair Encoding (BPE) tokenizer, an advanced causal language model built in PyTorch, and a dual-model client architecture capable of seamless context switching between general conversation and specialized code generation.

## 🌟 Features

*   **Custom Transformer Architecture**: Implements a GPT-style causal language model utilizing Multi-Head Attention, GeLU activations, and Layer Normalization.
*   **Dynamic BPE Tokenizer**: A fully custom tokenizer that trains directly from text data, dynamically adjusting its vocabulary size up to a configured maximum limit.
*   **Dual-Model Inference System**:
    *   **Model 1 (Chat)**: Handles general conversational context, reasoning, and memory.
    *   **Model 2 (Code)**: A specialized secondary model. When Model 1 generates the `<CONNECT_TO_CODER_M2>` trigger token, the prompt is automatically routed to Model 2 to generate high-quality Python code.
*   **Context Memory Management**: Retains conversational history up to a defined token limit (`MAX_HISTORY_TOKENS`), dynamically popping older turns to prevent exceeding the context window while maintaining conversation flow.
*   **Interrupt-Safe Training**: The training loop safely catches `KeyboardInterrupt` (Ctrl+C), ensuring that the latest model weights and optimizer states are saved to disk before shutting down.
*   **Hardware Optimized**: Leverages PyTorch 2.0+ Automatic Mixed Precision (`torch.amp.autocast` using `bfloat16`) and sets `torch.set_float32_matmul_precision('high')` for maximized training performance on modern CUDA hardware.
*   **Self-Healing Checkpoint Loading**: If a model checkpoint is loaded but the vocabulary size has changed, the inference script automatically detects the structural mismatch and rebuilds the embedding tables to safely load the weights.

## 📂 Project Structure

*   `train.py`: The core training script. Defines the `AdvancedLanguageModel` class (Transformer blocks), manages the training loop, gradient accumulation, and interval checkpoint saving.
*   `tokenizer_utils.py`: Contains the `DynamicBPETokenizer` class. Handles byte-pair encoding from scratch, token frequency counting, merging, and saving/loading the vocabulary maps.
*   `client.py` *(Main Entry)*: The interactive CLI application that handles user inputs, maintains dialogue history, dynamically builds context prompts, and orchestrates the handoff between the Chat and Code models.

## ⚙️ Prerequisites

*   Python 3.8+ (Optimized and tested for Python 3.12)
*   PyTorch 2.0+ (Required for `scaled_dot_product_attention` and `bfloat16`)
*   A CUDA-compatible GPU is highly recommended for reasonable training speeds.

## 🚀 Installation & Setup

1.  **Clone the repository and install dependencies**:
    ```bash
    pip install torch
    ```
2.  **Prepare Training Data**:
    Create a text file named `input_chatbot.txt` in the root directory. This file should contain your raw training dialogue formatted with `<|endofline|>` tokens separating individual conversational turns.

## 🧠 Training the Model

To begin training the primary chat model, run:
```bash
python train.py
```

**Default Model Hyperparameters:**
*   **Architecture**: 12 Layers, 8 Attention Heads, 512 Embedding Dimension.
*   **Training Loop**: Max iterations = 10,000, Batch Size = 2, Block Size = 2048, Gradient Accumulation = 32 steps.
*   **Checkpoints**: The model continuously saves primary weights to `chatbot_model.pth` and periodically saves interval backups into the `checkpoints_backup/` directory.

## 💬 Running the Client

Once training is complete (or if you already have pre-trained `.pth` weights), launch the chat interface:
```bash
python client.py
```

### How the Dual-Model Handoff Works
1.  The user types a prompt into the console. The client prepends previous conversation history.
2.  **Model 1 (Chat)** processes the full memory prompt.
3.  If the chat model determines that programming is required, it outputs a special trigger `<CONNECT_TO_CODER_M2>` followed by the coding instructions.
4.  The client intercepts this token. It verifies if `tokenizer_code.pkl` and `code_model.pth` exist.
5.  If found, **Model 2 (Coder)** is invoked with the instructions.
6.  The code generation output is formatted and presented to the user under the `⚡ [AI LẬP TRÌNH PHẢN HỒI]` banner.
