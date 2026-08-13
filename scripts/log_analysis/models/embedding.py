import logging
import time

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_embedder import LogVectorStore
from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult
from log_analysis.models.base import BaseModel
from log_analysis.models.generative import GenerativeConfig, GenerativeModel

logger = logging.getLogger(__name__)


class EmbeddingConfig(PydanticBaseModel):
    generative: GenerativeConfig
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    db_path: str = "./embeddings/log_embeddings.db"
    collection_name: str = "log_embeddings"
    similarity_threshold: float = 0.8
    hf_device: str = "cpu"
    token: str = None
    clearCollectionAtStartup: bool = False  # If True, delete the Milvus collection on startup (testing only)


class EmbeddingModel(BaseModel):
    """Classifies new log entries through a generative LLM, persists each classification
    as a vector in the Milvus store, and on subsequent runs answers similar log entries
    straight from the vector store without invoking the LLM again."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._encoder = None
        self._store = None
        self._fallback = GenerativeModel(config.generative)

    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading sentence embedding model %s on %s",
            self.config.embedding_model_name,
            self.config.hf_device,
        )
        kwargs = {"device": self.config.hf_device}
        if self.config.token:
            kwargs["token"] = self.config.token
        self._encoder = SentenceTransformer(self.config.embedding_model_name, **kwargs)
        get_dim = getattr(self._encoder, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._encoder.get_embedding_dimension  # type: ignore
        dimension = get_dim()
        self._store = LogVectorStore(
            self.config.db_path,
            collection_name=self.config.collection_name,
            dimension=dimension,
        )
        if self.config.clearCollectionAtStartup:
            logger.warning("Clearing Milvus collection %s at startup (testing only)", self.config.collection_name)
            self._store.client.drop_collection(self.config.collection_name)
            self._store._ensure_collection()

    def _store_safe(self) -> LogVectorStore:
        if self._store is None:
            self._load_encoder()
        return self._store

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is None:
            self._load_encoder()
        vectors = self._encoder.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        batch_text = "\n".join(f"[{entry.index}] {entry.raw_text}" for entry in batch.entries)
        start = time.perf_counter()
        [vector] = self._embed_batch([batch_text])
        logger.info("Embedded batch #%d in %.3fs", batch.batch_id, time.perf_counter() - start)

        matches = self._store_safe().search_batch([vector], top_k=1)
        hits = matches[0] if matches else []
        distance = hits[0].get("distance", 0.0) if hits else 0.0

        if hits and distance >= self.config.similarity_threshold:
            print(f"Reusing cached classification for batch #{batch.batch_id} (distance={distance:.4f})")
            entity = hits[0].get("entity", {})
            return BatchAnalysisResult(
                batch_id=batch.batch_id,
                error_found=bool(entity.get("is_error", False)),
                error_count=1 if entity.get("is_error", False) else 0,
                results=[],
            )

        logger.info("Classifying batch #%d with LLM", batch.batch_id)
        fallback_result = self._fallback.analyze(batch)
        self._store_safe().insert(
            vector,
            batch_text,
            fallback_result.error_found,
            None,
            None,
        )

        return BatchAnalysisResult(
            batch_id=batch.batch_id,
            error_found=fallback_result.error_found,
            error_count=fallback_result.error_count,
            results=[],
        )

    ##Droping database ONLY FOR TESTING PURPOSES
    def _delete_collection(self) -> None:
        self._store_safe().client.drop_collection(self.config.collection_name)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()