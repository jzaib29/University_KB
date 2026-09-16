import json
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, config_path="vector_db/config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config["embedding_model"]

        self.normalize = self.config.get(
            "normalized_embeddings",
            True
        )

        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )

        return np.asarray(
            embedding,
            dtype="float32"
        )
