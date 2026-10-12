import tempfile
import unittest
from pathlib import Path

from log_analysis.core.log_embedder import LogVectorStore
from log_analysis.core.log_entry import BatchAnalysisResult, LogBatch, LogEntry
from log_analysis.models.embedding import EmbeddingConfig, EmbeddingModel
from log_analysis.models.generative import GenerativeConfig

BATCH = LogBatch(batch_id=0, entries=[LogEntry(index=0, source_file="a.log", raw_text="ERROR")])


class _FakeVectorStore:
    """Stand-in for the Milvus store: canned hits, recorded inserts."""

    def __init__(self, hits=None) -> None:
        self.hits = hits
        self.inserted: list[dict] = []

    def search_batch(self, vectors, top_k: int = 1):
        return [self.hits] if self.hits else [[] for _ in vectors]

    def insert(self, vector, raw_text, is_error, **fields):
        self.inserted.append({"is_error": is_error, **fields})


class _StubFallback:
    """Stand-in for the generative model, counting the calls the cache is supposed to avoid."""

    def __init__(self, result: BatchAnalysisResult) -> None:
        self.result = result
        self.calls = 0

    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        self.calls += 1
        return self.result


class _StubEmbeddingModel(EmbeddingModel):
    """EmbeddingModel whose encoder, vector store and LLM call are all stubs."""

    def __init__(self, store: _FakeVectorStore, fallback: _StubFallback) -> None:
        self.config = EmbeddingConfig(
            generative=GenerativeConfig(model_name="stub-model", backend="huggingface", contextWindow=4096)
        )
        self._encoder = object()
        self._store = store
        self._fallback = fallback

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


def _hit(probability=0.87, **entity_overrides) -> dict:
    entity = {"is_error": True, "model_name": "stub-model", "embedder_model_name": "stub-embedder"}
    if probability is not None:
        entity["error_probability"] = probability
    entity.update(entity_overrides)
    return {"distance": 0.93, "entity": entity}


class CachedProbabilityTest(unittest.TestCase):
    def test_a_cache_hit_reports_the_probability_stored_with_the_match(self):
        store = _FakeVectorStore(hits=[_hit(0.87)])
        fallback = _StubFallback(BatchAnalysisResult(batch_id=0, error_found=True, is_valid_response=True))
        model = _StubEmbeddingModel(store, fallback)

        result = model.analyze(BATCH)

        self.assertEqual(result.embedding_hit, True)
        self.assertAlmostEqual(result.error_probability, 0.87)
        self.assertEqual(fallback.calls, 0)  # the LLM was never needed
        self.assertEqual(store.inserted, [])

    def test_a_cache_hit_without_a_stored_score_stays_unknown(self):
        store = _FakeVectorStore(hits=[_hit(None)])
        model = _StubEmbeddingModel(store, _StubFallback(BatchAnalysisResult(batch_id=0, error_found=True)))

        result = model.analyze(BATCH)

        self.assertEqual(result.embedding_hit, True)
        self.assertIsNone(result.error_probability)

    def test_a_miss_stores_the_fresh_probability_for_later_hits(self):
        store = _FakeVectorStore(hits=None)
        fallback = _StubFallback(
            BatchAnalysisResult(
                batch_id=0, error_found=True, is_valid_response=True, error_probability=0.3,
                model_name="stub-model",
            )
        )
        model = _StubEmbeddingModel(store, fallback)

        result = model.analyze(BATCH)

        self.assertEqual(fallback.calls, 1)
        self.assertAlmostEqual(result.error_probability, 0.3)
        self.assertEqual(len(store.inserted), 1)
        self.assertAlmostEqual(store.inserted[0]["error_probability"], 0.3)

    def test_an_invalid_response_is_not_cached(self):
        store = _FakeVectorStore(hits=None)
        fallback = _StubFallback(
            BatchAnalysisResult(batch_id=0, error_found=False, is_valid_response=False, error_probability=None)
        )
        model = _StubEmbeddingModel(store, fallback)

        model.analyze(BATCH)

        self.assertEqual(store.inserted, [])

    def test_the_gpu_sentinel_becomes_a_torch_device_string(self):
        model = _StubEmbeddingModel(_FakeVectorStore(), _StubFallback(BatchAnalysisResult(batch_id=0)))

        model.config.hf_device = "gpu"
        self.assertEqual(model._encoder_device(), "cuda")
        model.config.hf_device = "cpu"
        self.assertEqual(model._encoder_device(), "cpu")


class CacheSchemaTest(unittest.TestCase):
    def test_a_cache_written_before_the_score_field_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cache.db"
            self._write_legacy_collection(db_path)

            store = LogVectorStore(db_path, collection_name="log_embeddings", dimension=4)
            store.insert([1.0, 0.0, 0.0, 0.0], "new entry", True, model_name="m", error_probability=0.42)
            hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
            store.close()

            self.assertEqual(len(hits), 1)
            self.assertAlmostEqual(hits[0]["entity"]["error_probability"], 0.42)

    @staticmethod
    def _write_legacy_collection(db_path: Path) -> None:
        """Create the pre-error_probability schema the shipped cache used."""
        from pymilvus import DataType, MilvusClient

        client = MilvusClient(str(db_path))
        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("raw_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("is_error", DataType.BOOL)
        index_params = client.prepare_index_params()
        index_params.add_index("vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection("log_embeddings", schema=schema)
        client.create_index("log_embeddings", index_params)
        client.load_collection("log_embeddings")
        client.insert("log_embeddings", [{"vector": [1.0, 0.0, 0.0, 0.0], "raw_text": "legacy", "is_error": True}])
        client.close()


if __name__ == "__main__":
    unittest.main()
