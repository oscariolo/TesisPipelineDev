import logging
from pathlib import Path
from typing import Optional

from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)


class LogVectorStore:
    """Milvus Lite-backed vector store that persists classified log groups to a local .db file."""

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str = "log_embeddings",
        dimension: int = 384,
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.dimension = dimension
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(str(self.db_path))
        self._ensure_collection()
        self.client.load_collection(self.collection_name)

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return
        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field("raw_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("is_error", DataType.BOOL)
        schema.add_field("error_description", DataType.VARCHAR, max_length=4096, nullable=True)
        schema.add_field("recommended_action", DataType.VARCHAR, max_length=4096, nullable=True)
        index_params = self.client.prepare_index_params()
        index_params.add_index("vector", index_type="FLAT", metric_type="COSINE")
        self.client.create_collection(self.collection_name, schema=schema)
        self.client.create_index(self.collection_name, index_params)
        logger.info("Created collection %s in %s", self.collection_name, self.db_path)

    def search(
        self,
        vector: list,
        top_k: int = 1,
    ) -> Optional[list[dict]]:
        """Return nearest neighbors as [{id, distance, entity}], None if the collection is
        empty or no hit was found."""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["raw_text", "is_error", "error_description", "recommended_action"],
        )
        hits = results[0] if results else []
        return hits or None

    def search_batch(
        self,
        vectors: list[list[float]],
        top_k: int = 1,
    ) -> list[list[dict]]:
        if not vectors:
            return []
        results = self.client.search(
            collection_name=self.collection_name,
            data=vectors,
            limit=top_k,
            output_fields=["raw_text", "is_error", "error_description", "recommended_action"],
        )
        return results if results else [[] for _ in vectors]

    def insert(
        self,
        vector: list,
        raw_text: str,
        is_error: bool,
        error_description: Optional[str],
        recommended_action: Optional[str],
    ) -> None:
        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": vector,
                    "raw_text": raw_text,
                    "is_error": is_error,
                    "error_description": error_description or None,
                    "recommended_action": recommended_action or None,
                }
            ],
        )
    def close(self) -> None:
        self.client.close()