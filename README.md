#🧠 ProximaGPT

A tiny language model built from scratch with Python & NumPy.

ProximaGPT is an educational from-scratch language model project.
The goal is to understand how the fundamental pieces of a language model work by building them step by step—without relying on high-level ML frameworks.

---

✨ v0.1 — Foundation

The first version implements the basic learning pipeline:

Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
Mean Pooling
 ↓
Linear Layer
 ↓
Softmax
 ↓
Loss
 ↓
Gradient
 ↓
Weight Update
 ↓
Prediction

What's included

- 🔤 Basic tokenizer
- 🧩 Token embeddings
- 🧠 Neural network forward pass
- 📊 Softmax probability distribution
- 📉 Cross-entropy loss
- 🔬 Numerical gradient
- 🔄 Training loop
- 💾 Model save & load
- 🤖 Next-token prediction

---
```
📂 Project Structure 

ProximaGPT/
│
├── README.md
│
├── v0.1-foundation/
│   ├── neural_network.py
│   ├── tokenizer.py
│   ├── training_data.py
│   └── sunnygpt_model.npz
│
├── docs/
│   └── v0.1_notes.md
│
└── .gitignore
```
---

🧪 Example

Enter text: I am

Input IDs: [1, 6]

Predicted word: love

This is a very small experimental model, so its predictions are limited to the patterns present in its tiny training dataset.

---

🛠️ Built With

- Python
- NumPy
- Termux
- No high-level machine-learning framework

---

🚀 Roadmap

v0.1  Foundation        ✅
v0.2  Backpropagation   🔜
v0.3  Better Training   🔜
v0.4  Attention         🔜
v0.5  Transformer       🔜
v1.0  Mini-GPT          🔜

The long-term goal is to build a small Transformer-based Mini-GPT from scratch and learn the architecture by implementing each component personally.

---

📖 Why ProximaGPT?

This project is not about building the biggest model.

It's about understanding how the model works.

«Build it. Break it. Understand it. Improve it.»

---

👨‍💻 Project Status

v0.1 Foundation — Complete ✅



![image alt](https://github.com/sunnysarwar671-boop/learning_journey/blob/aace831ff0e8ea9ac59e2582f0319ca2559fba44/file_00000000053c81fabdfe6dbc3636db8f.png)
