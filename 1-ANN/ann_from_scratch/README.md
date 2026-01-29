#  Neural Networks From Scratch — Two Complementary Implementations

This folder contains **two independent implementations of a Multi-Layer Perceptron (MLP) built entirely from scratch**, without using deep learning frameworks such as PyTorch or TensorFlow.

Although both implementations solve similar problems, they come from **two different pedagogical traditions** and are intentionally kept together for comparison and deeper understanding.

---

##  Implementation 1  
### Deep Learning Specialization — Andrew Ng (Coursera)

**Files**
- `1.09 Building Neural Network from scratch.ipynb`
- `Building your Deep Neural Network Step by Step.ipynb`
- `dnn_utils.py`
- `testCases.py`
- `public_tests.py`
- `test_utils.py`

**Origin**  
These notebooks are inspired by the **Deep Learning Specialization by Andrew Ng**, one of the most respected and rigorous introductions to deep learning worldwide.

**Why this implementation is special**
- Extremely **rigorous and well-structured**
- Fully **mathematical and vectorized**
- Implements:
  - Forward propagation
  - Backpropagation
  - Gradient descent
- Designed to **scale to an arbitrary number of layers (n-layer networks)**
- Clear separation of:
  - Parameters
  - Caches
  - Gradients
- Includes **unit tests** to validate correctness at every step

**Pedagogical Value**
> This is one of the **cleanest and most rigorous MLP implementations** you will find for learning *how deep learning actually works internally*.

It teaches you:
- How modern deep learning frameworks are designed
- How backpropagation is implemented in practice
- How to think in terms of computational graphs and vectorized math

---

## Implementation 2  
### *Hands-On Deep Learning Algorithms with Python*

**Files**
- Alternative notebook implementation (XOR-focused)

**Origin**  
This implementation is inspired by the book:  
**_Hands-On Deep Learning Algorithms with Python_**

**Philosophy**
- More **intuitive and conceptual**
- Less abstraction, more step-by-step logic
- Focuses on understanding *why* neural networks work

**Key Characteristics**
- Solves the classic **XOR problem**
- Closely tied to the **historical origins of the MLP**
- Easier to follow for first-time learners
- Emphasizes intuition over scalability

**Pedagogical Value**
> The XOR problem is historically important because it demonstrated the **limitations of single-layer perceptrons** and motivated the invention of multi-layer neural networks.

This implementation helps you understand:
- Why hidden layers are necessary
- How non-linearity emerges
- How neural networks first appeared in machine learning history

---

##  Why Keep Both Implementations?

| Andrew Ng Version | Book Version |
|------------------|-------------|
| Highly rigorous | Highly intuitive |
| Scales to **n layers** | Focuses on small networks |
| Production-style design | Conceptual & historical |
| Test-driven | Easy to modify & experiment |

Together, they provide:
- **Mathematical rigor**
- **Historical context**
- **Engineering intuition**

This combination gives a **complete mental model** of neural networks — from their origins to their modern implementations.

---

## 🎯 Recommendation for Learners

- Start with the **XOR-based implementation** to build intuition
- Then study the **Andrew Ng implementation** to understand how deep learning is engineered at scale
- Compare both to solidify your understanding of:
  - Backpropagation
  - Layer-wise abstraction
  - Model scalability

---

## 📜 Credits & Attribution

- **Deep Learning Specialization** — Andrew Ng (Coursera)
- **Hands-On Deep Learning Algorithms with Python** — Open-source educational material

These implementations are used **strictly for educational purposes**.

---

> *Understanding neural networks from scratch is the fastest path to mastering modern AI frameworks.*


