🧠 Small Language Model (SLM) - From Scratch
This project demonstrates how to build, pre-train, and run inference on a small transformer-based language model using the TinyStories dataset. It closely follows a minimalistic approach inspired by Karpathy’s nanoGPT and is implemented using Python and PyTorch.

🧩 Project Structure
The entire training pipeline is broken down into intuitive steps to demystify the inner workings of transformer-based language models.

✅ Steps Covered
Load Dataset: Uses the TinyStories dataset from Hugging Face.

Tokenization: Efficiently encodes text using tiktoken and saves it to disk (train.bin, val.bin) to conserve RAM.

Batch Creation: Creates input-output batches for training.

Model Architecture: Defines a transformer-based Small Language Model (SLM).

Loss Function: Uses cross-entropy loss for next-token prediction.

Training Configuration: Splits into optimizer, scheduler, and gradient steps.

Pretraining: Trains the model over tokenized sequences.

Visualization: Plots training loss to monitor learning.

Inference: Generates text using the trained SLM.

📦 Installation
Install the required dependencies:

bash
Copy
Edit
pip install -U datasets tiktoken torch numpy tqdm matplotlib
📚 Dataset
Name: TinyStories

Source: roneneldan/TinyStories

Description: Short, simple, and syntactically correct English stories ideal for training small language models.

🧪 Training Workflow
1. Load and Preview Dataset
python
Copy
Edit
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
2. Tokenization
python
Copy
Edit
import tiktoken
enc = tiktoken.get_encoding("gpt2")
Each text sample is encoded using GPT-2's tokenizer and stored efficiently in .bin files.

3. Batch Loader
Creates batched input-target pairs using a memory-efficient strategy to feed the transformer model.

4. Model Architecture
Implements a simplified transformer-based model with:

Positional Embeddings

Multi-head Self-Attention

Feed-forward layers

Layer Normalization

5. Training Loop
Optimizes the model using AdamW and records the loss during training.

📈 Results
The loss curve is plotted after training to monitor convergence.

The trained model is capable of generating simple English sentences.

💡 Inference Example
python
Copy
Edit
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = model.generate(context, max_new_tokens=100)
🧠 Model Capabilities
Learns next-token predictions from scratch.

Can generate TinyStories-style narratives.

Minimal compute requirement – can be trained on consumer GPUs (e.g., NVIDIA T4, A100).

📂 Files Generated
train.bin / val.bin – Preprocessed and tokenized binary dataset.

slm.pt – Trained PyTorch model weights.

loss.png – Plot of training loss over iterations.

✍️ Author
Subhajeet Krishna Dey
AI/ML enthusiast passionate about small models, transformers, and coding from scratch.
GitHub: @subho0505

📜 License
This project is open-sourced under the MIT License.

