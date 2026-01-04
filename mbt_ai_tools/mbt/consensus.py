from typing import Iterable, List, Optional, Tuple

import numpy as np

from .embeddings import embed_texts
from .geometry import geometric_median, shock
from .stability import internal_entropy
from .utils import UniversalAIClient, build_variation_prompt


class ManifoldRegulator:
    """
    Minimal manifold regulator mirroring the MBT-5 core loop.

    It preserves the Weiszfeld median calculation and squared-distance shock
    measurement from the Colab demos.
    """

    def __init__(self):
        self._manifold_center: Optional[np.ndarray] = None

    def set_manifold(self, samples: Iterable[str]) -> None:
        samples = list(samples)
        if not samples:
            return

        embeddings = embed_texts(samples)
        if len(samples) == 1:
            self._manifold_center = embeddings[0]
            return

        self._manifold_center = geometric_median(embeddings)

    def shock(self, text: str) -> float:
        if self._manifold_center is None:
            return 0.0

        embedding = embed_texts([text])[0]
        return shock(embedding, self._manifold_center)


class ZeroBoxConsensus:
    """
    Self-consistency pilot for generating multiple answers and scoring entropy.

    The sampling logic mirrors the notebook: three answer variations are
    requested with minimal prompt decoration.
    """

    def __init__(self, ai_client: Optional[UniversalAIClient] = None):
        self.ai = ai_client or UniversalAIClient()

    def sample_answers(self, prompt: str, system_prompt: str) -> List[str]:
        samples: List[str] = []
        for i in range(3):
            p = build_variation_prompt(prompt, i)
            samples.append(self.ai.call_ai(p, system_prompt))
        return samples

    def evaluate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Answer clearly and directly.",
    ) -> Tuple[str, float]:
        responses = self.sample_answers(prompt, system_prompt)
        entropy = internal_entropy(responses)
        return responses[0], entropy
