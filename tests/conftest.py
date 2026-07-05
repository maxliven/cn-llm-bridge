"""Test configuration and shared fixtures for cn-llm-bridge."""

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Ensure tests use mock API keys, not real ones."""
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    monkeypatch.setenv("KIMI_MODEL", "kimi-k2.7-code")
    yield


@pytest.fixture
def sample_vision_response():
    """Mock Qwen vision API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"summary": "test summary", '
                        '"details": "test details", '
                        '"text_found": "sample text", '
                        '"objects": ["button", "text field"]}'
                    )
                }
            }
        ]
    }


@pytest.fixture
def sample_kimi_response():
    """Mock Kimi chat API response."""
    return {"choices": [{"message": {"content": "Kimi analysis result: the code is well-structured."}}]}
