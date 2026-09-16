import json
import faiss

from src.embeddings import EmbeddingService


class FAISSRetriever:
    def __init__(
        self,
        index_path="vector_db/faiss_index.bin",
        metadata_path="vector_db/metadata.json",
        config_path="vector_db/config.json"
    ):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embedding model
        self.embedding_service = EmbeddingService(
            config_path=config_path
        )

        # Safety check
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS index contains {self.index.ntotal} vectors "
                f"but metadata contains {len(self.metadata)} chunks."
            )

    def search(self, query: str, k: int = 5):
        query_embedding = (
            self.embedding_service.embed_query(query)
        )

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            item = self.metadata[idx].copy()

            item["score"] = float(score)

            results.append(item)

        return results
