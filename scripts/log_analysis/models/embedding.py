import logging
import time

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_embedder import LogVectorStore
from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult, render_batch_text
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
    # False: still query the store (and keep saving to it) but always let the LLM answer.
    use_embedding_cache: bool = True
    clearCollectionAtStartup: bool = False  # If True, delete the Milvus collection on startup (testing only)


class EmbeddingModel(BaseModel):
    """Classifies new log entries through a generative LLM, persists each classification
    as a vector in the Milvus store, and on subsequent runs answers similar log entries
    straight from the vector store without invoking the LLM again."""

    # The generative path's default device string; sentence-transformers rejects it.
    DEVICE_ALIASES = {"gpu": "cuda"}

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._encoder = None
        self._store = None
        self._fallback = GenerativeModel(config.generative)

    def _encoder_device(self) -> str:
        """Torch device string for the encoder (``main.py`` shares ``--hf-device`` with the LLM)."""
        device = self.config.hf_device or "cpu"
        return self.DEVICE_ALIASES.get(device.lower(), device)

    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading sentence embedding model %s on %s",
            self.config.embedding_model_name,
            self._encoder_device(),
        )
        kwargs = {"device": self._encoder_device()}
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
        batch_text = render_batch_text(batch)
        start = time.perf_counter()
        [vector] = self._embed_batch([batch_text])
        logger.info("Embedded batch #%d in %.3fs", batch.batch_id, time.perf_counter() - start)

        matches = self._store_safe().search_batch([vector], top_k=1)
        hits = matches[0] if matches else []
        distance = hits[0].get("distance", 0.0) if hits else 0.0
        matched = bool(hits) and distance >= self.config.similarity_threshold

        if matched and self.config.use_embedding_cache:
            logger.info(
                "Batch #%d matched existing embedding with distance %.4f, skipping LLM",
                batch.batch_id,
                distance,
            )
            entity = hits[0].get("entity", {})
            return BatchAnalysisResult(
                batch_id=batch.batch_id,
                error_found=bool(entity.get("is_error", False)),
                # The score belongs to the entry this batch matched, not to a fresh model call.
                error_probability=entity.get("error_probability"),
                model_name=entity.get("model_name", None),
                embedder_model_name=entity.get("embedder_model_name", None),
                is_valid_response=True, #if it was gotte from embedding store its assumed valid
                embedding_hit=True,
                similarity=distance,
            )

        if matched:
            logger.info(
                "Batch #%d matched existing embedding with distance %.4f but the cache is ignored",
                batch.batch_id,
                distance,
            )

        logger.info("Classifying batch #%d with LLM", batch.batch_id)
        return self._classify_with_llm(batch, vector, batch_text, distance)

    def _classify_with_llm(self, batch: LogBatch, vector: list[float], batch_text: str,
                           similarity) -> BatchAnalysisResult:
        """Classify with the LLM and remember the embedding for later hits.

        ``embedding_hit`` stays False here (the answer did not come from the cache) even when the
        lookup matched and the cache is simply being ignored; ``similarity`` keeps the measured
        distance either way.
        """
        fallback_result = self._fallback.analyze(batch)
        fallback_result.embedder_model_name = self.config.embedding_model_name
        fallback_result.embedding_hit = False
        fallback_result.similarity = similarity
        if fallback_result.is_valid_response: #solo guarda el embedding si la respuesta del modelo es valida
            self._store_safe().insert(
                vector,
                batch_text,
                fallback_result.error_found,
                embedder_model_name=fallback_result.embedder_model_name,
                model_name=fallback_result.model_name,
                error_probability=fallback_result.error_probability,
            )

        return fallback_result

    ##Droping database ONLY FOR TESTING PURPOSES
    def _delete_collection(self) -> None:
        self._store_safe().client.drop_collection(self.config.collection_name)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()