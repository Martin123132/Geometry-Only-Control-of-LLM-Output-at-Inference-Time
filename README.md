# 🛡️ MBT-5 Reality Guardian  
### A Geometric Hallucination Regulator for Large Language Models

**MBT-5** (Manifold-Based Trajectory v5) is a semantic regulation layer for LLMs.  
It detects and suppresses hallucinations by measuring **geometric drift** between generated text and a user-defined reference manifold.

This system does not rely on rules, classifiers, or safety policies.  
It operates purely on **embedding geometry**.

---

## What MBT-5 Does

MBT-5 treats language generation as motion through a high-dimensional semantic space.

It:

- Constructs a **reference manifold** from user-provided texts
- Measures **semantic distance** between model output and that manifold
- Interprets large deviations as **hallucinations or domain drift**
- Actively **corrects or suppresses** misaligned outputs
- Refuses to emit output when alignment cannot be recovered

In practice, this allows you to **constrain an LLM to a specific technical, legal, or stylistic reality** without rewriting prompts or relying on behavioral instructions.

---

## Core Concepts

### 1. Semantic Manifold
A manifold is defined by 3–10 short reference texts representing the domain you want the model to stay within (e.g. CUDA kernels, Rust memory safety, legal filings).

The manifold center is computed as the **geometric (L1) median** of the embeddings.

---

### 2. Shock
Shock is the squared Euclidean distance between a generated response and the manifold center:

shock = || embedding(output) − manifold_center ||²

Low shock = output belongs to the domain  
High shock = semantic drift / hallucination

---

### 3. Recursive Healing
If shock exceeds a threshold:

1. The output is rejected
2. Feedback is generated describing the drift
3. The model is re-queried
4. Shock is re-evaluated

This continues until:
- Alignment is recovered, or
- A hard rejection threshold is reached

---

## Why This Works

Hallucinations are not random errors.  
They are **geometric departures** from the semantic neighborhood implied by the task.

MBT-5 does not attempt to decide what is “true”.  
It enforces **domain coherence**.

If a model cannot produce an answer that belongs to the declared manifold, **it is prevented from speaking**.

---

## Example Use Cases

- Preventing speculative answers in technical domains
- Suppressing “helpful” but incorrect explanations
- Enforcing professional tone without prompt engineering
- Detecting policy-driven or stylistic drift
- Auditing LLM outputs for domain violations

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MBT-5-Reality-Guardian.git
cd MBT-5-Reality-Guardian
pip install -r requirements.txt


⸻

Running the Pilot

Run pilot_app.py inside Jupyter or VS Code.
	1.	Paste your API key (never stored)
	2.	Define a manifold (3–5 domain lines)
	3.	Lock reality
	4.	Query the model
	5.	Observe shock, correction, or rejection

⸻

Repository Structure

MBT-5-Reality-Guardian/
├── mbt5_core.py        # Geometry + embedding logic
├── pilot_app.py        # Interactive UI + recursive pilot
├── examples/           # Sample manifolds
├── requirements.txt
└── README.md


⸻

Mathematical Summary
	•	Embeddings: all-MiniLM-L6-v2 (384-D)
	•	Manifold center: L1 geometric median
	•	Drift metric: squared L2 distance
	•	Control method: recursive rejection + regeneration

This is a control system, not a classifier.


