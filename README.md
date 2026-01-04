# MBT-5 (Geometry-Only, Inference-Time Regulator)

MBT-5 is a geometry-only, inference-time regulator for large language models. It estimates output confidence and hallucination risk by measuring semantic stability and geometric drift in embedding space.

- **No training**
- **No fine-tuning**
- **No classifiers**

The package keeps the original MBT-5 math intact: internal entropy across multiple responses, manifold shock via squared distance, and leave-one-out token shocks.

## Installation

```bash
pip install mbt-ai-tools
```

## Usage

### Library

```python
from mbt_ai_tools import confidence_score, hallucination_risk, token_shock_map

text = "The capital of France is Paris."

score = confidence_score(text)         # Internal entropy (lower = more confident)
risk = hallucination_risk(text)        # {'score': ..., 'label': ..., 'color': ...}
shocks = token_shock_map(text)         # [('The', ...), ('capital', ...), ...]
```

- To mirror the original Zero-Box pilot, pass multiple blank-line separated responses in a single string and MBT-5 will compute the same internal entropy used in the notebooks.
- For manifold-based measurements, use `ManifoldRegulator` to set a semantic manifold and measure squared shock.

### CLI

```bash
mbt-check "The capital of France is Paris."
```

Outputs a confidence label and numeric internal entropy score. No network calls are made unless you explicitly configure an AI client in your own code.

## Project Layout

```
mbt_ai_tools/
├── mbt/
│   ├── embeddings.py      # SentenceTransformer loader
│   ├── geometry.py        # geometric median, shock, distance
│   ├── stability.py       # self-consistency / entropy scoring
│   ├── tokens.py          # leave-one-out token shock
│   ├── consensus.py       # multi-agent / council logic
│   └── utils.py
├── cli.py                 # optional CLI: mbt-check "text"
├── pyproject.toml
├── README.md
└── LICENSE
```

## Notes

- MBT-5 is inference-time only. It does not alter model weights or training data.
- The package avoids moderation, safety rules, or content inspection. It measures geometric stability only.
- `ipywidgets` and other UI dependencies are not required at install time.
