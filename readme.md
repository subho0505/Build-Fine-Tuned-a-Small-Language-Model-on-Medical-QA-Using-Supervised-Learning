## Small Language Model (SLM) Project Documentation

### Overview
This project implements a transformer-based Small Language Model (SLM) trained on the TinyStories dataset for general text generation, followed by medical domain fine-tuning using the Medical-O1-Reasoning dataset. Key features include:

- GPT-2 architecture implementation
- Memory-efficient dataset handling with memmap
- Mixed-precision training
- Customizable model configuration
- Medical domain fine-tuning
- Text generation capabilities

### Project Structure
```
small_language_model.py
├── Dataset Preparation
├── Model Architecture
├── Training Configuration
├── Training Loop
├── Fine-tuning
└── Inference
```

---

### 1. Dataset Preparation
**TinyStories Dataset**
- Source: Hugging Face `roneneldan/TinyStories`
- Preprocessing:
  - Tokenization using GPT-2 tokenizer
  - Memory-mapped storage (`train.bin`/`validation.bin`)
- Characteristics:
  - Simple children's stories
  - Vocabulary size: 50,257 tokens

**Medical Dataset (Fine-tuning)**
- Source: `FreedomIntelligence/medical-o1-reasoning-SFT`
- Formatting:
  ```python
  "### Question:\n{Question}\n### Response:\n{Response}"
  ```
- 5% validation split

---

### 2. Model Architecture (GPT-2 Style)
```python
@dataclass
class GPTConfig:
    block_size: int  # Context length (128/512)
    vocab_size: int   # 50,257
    n_layer: int      # Number of transformer blocks
    n_head: int       # Attention heads
    n_embd: int       # Embedding dimension
```

**Key Components**:
1. **Embeddings**:
   - Token (`wte`) + Position (`wpe`) embeddings
   - Weight tying with output layer

2. **Transformer Block**:
   ```python
   Block(
     LayerNorm -> CausalSelfAttention -> LayerNorm -> MLP
   )
   ```
   - Multi-head attention with flash optimization
   - GELU-activated MLP (4x expansion)

3. **Output Head**:
   - Linear layer projecting to vocabulary space

---

### 3. Training Configuration
**Hyperparameters**:
```python
batch_size = 32
block_size = 128       # Increased to 512 for medical fine-tuning
learning_rate = 1e-4
max_iters = 40,000
gradient_accumulation_steps = 32
```

**Optimization**:
- AdamW with cosine decay schedule
- Linear warmup (1,000 steps)
- Gradient clipping (max_norm=0.5)
- Mixed precision training (`bfloat16`/`float16`)

**Hardware Setup**:
- Automatic device detection (CUDA/CPU)
- Memory-pinned tensors for GPU transfer

---

### 4. Training Workflow
```mermaid
graph LR
A[Load Dataset] --> B[Tokenize]
B --> C[Memmap Storage]
C --> D[Initialize Model]
D --> E[Training Loop]
E --> F[Batch Generation]
F --> G[Forward Pass]
G --> H[Loss Calculation]
H --> I[Backward Pass]
I --> J[Parameter Update]
```

**Key Functions**:
1. `get_batch()`: Generates input-target pairs
2. `estimate_loss()`: Evaluates train/validation loss
3. Checkpointing: Saves best model based on validation loss

**Training Output**:
- Periodic loss reporting (every 500 steps)
- Learning rate scheduling
- Loss curve visualization

---

### 5. Fine-tuning Process
**Medical Dataset Adaptation**:
1. Format conversion to Q&A pairs
2. Special token-free tokenization
3. Filtering empty sequences

**Fine-tuning Changes**:
- Increased context size (`block_size=512`)
- Reduced training iterations
- Domain-specific prompt format:
  ```
  ### Question:
  What causes fever?
  ### Response:
  Fever can be caused by...
  ```

---

### 6. Inference & Text Generation
**Generation API**:
```python
generate(idx, max_new_tokens, temperature=1.0, top_k=None)
```

**Example Usage**:
```python
prompt = "What are COVID-19 symptoms?"
context = torch.tensor(enc.encode_ordinary(prompt))
output = model.generate(context, max_new_tokens=100)
print(enc.decode(output[0].tolist()))
```

**Medical Output Example**:
```
### Question:
What are COVID-19 symptoms?
### Response:
Common symptoms include fever, cough, and shortness of breath...
```

---

### 7. Results & Evaluation
**Performance Metrics**:
- Final training loss: ~1.8
- Validation loss: ~2.1
- Loss curves show no severe overfitting

**Generation Quality**:
- Coherent short-story generation
- Basic medical question answering
- Limited reasoning capabilities (expected for model size)

---

### 8. Setup & Execution
**Dependencies**:
```bash
pip install torch datasets tiktoken tqdm matplotlib
```

**Execution Steps**:
1. Run script: `python small_language_model.py`
2. Automatic dataset download
3. Training starts automatically
4. Checkpoints saved to `best_model_params.pt`

---

### 9. Limitations & Improvements
**Current Limitations**:
- Small model capacity (6 layers, 384 dim)
- Limited context window (512 tokens)
- Basic medical knowledge only

**Improvement Paths**:
1. Scale up model dimensions
2. Longer training with larger batches
3. Instruction fine-tuning
4. RAG integration for medical accuracy
5. Quantization for deployment

---

### 10. References
1. TinyStories Dataset: [Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories)
2. Medical Dataset: [Medical-O1-Reasoning](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
3. NanoGPT Inspiration: [GitHub](https://github.com/karpathy/nanoGPT)
4. PyTorch Mixed Precision: [Official Docs](https://pytorch.org/docs/stable/amp.html)

> **Note**: This implementation prioritizes educational clarity over production-grade optimizations. For deployment scenarios, consider model quantization, ONNX conversion, and serving frameworks like FastAPI.