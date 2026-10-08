import pytest

from rag_engine.config import ChunkingConfig, ChunkStrategy, Settings


def test_defaults_are_valid() -> None:
    settings = Settings.from_env({})
    assert settings.chunking.strategy is ChunkStrategy.STRUCTURE
    assert settings.chunking.chunk_overlap < settings.chunking.chunk_size


def test_env_overrides_are_parsed_and_typed() -> None:
    settings = Settings.from_env(
        {
            "RAG_CHUNK_STRATEGY": "fixed",
            "RAG_CHUNK_SIZE": "128",
            "RAG_CHUNK_OVERLAP": "0",
            "RAG_INCLUDE_HEADING_CONTEXT": "false",
            "RAG_MAX_FILE_BYTES": "1000",
            "UNRELATED": "ignored",
        }
    )
    assert settings.chunking.strategy is ChunkStrategy.FIXED
    assert settings.chunking.chunk_size == 128
    assert settings.chunking.chunk_overlap == 0
    assert settings.chunking.include_heading_context is False
    assert settings.ingestion.max_file_bytes == 1000


def test_blank_env_values_fall_back_to_defaults() -> None:
    assert Settings.from_env({"RAG_CHUNK_SIZE": "  "}).chunking.chunk_size == 256


@pytest.mark.parametrize(
    "env",
    [
        {"RAG_CHUNK_SIZE": "64", "RAG_CHUNK_OVERLAP": "64"},
        {"RAG_CHUNK_SIZE": "not-a-number"},
        {"RAG_CHUNK_STRATEGY": "semantic-magic"},
        {"RAG_CHUNK_SIZE": "32", "RAG_MIN_CHUNK_TOKENS": "40"},
    ],
)
def test_invalid_env_raises_readable_error(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="invalid RAG_"):
        Settings.from_env(env)


def test_config_is_immutable() -> None:
    cfg = ChunkingConfig()
    with pytest.raises(ValueError):
        cfg.chunk_size = 10  # type: ignore[misc]
