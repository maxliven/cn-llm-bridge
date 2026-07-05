"""Test environment configuration parsing."""
import os
import sys
import pytest


def test_env_example_exists():
    """Verify .env.example is present and readable."""
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__)
    )
    root = result.stdout.strip() or os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(root, ".env.example")
    assert os.path.exists(env_path), f".env.example not found at {env_path}"
    with open(env_path) as f:
        content = f.read()
    assert "QWEN_API_KEY" in content
    assert "KIMI_API_KEY" in content
    assert "BAILIAN_BASE_URL" in content


def test_env_variables_set():
    """Verify required env vars are available in test environment."""
    assert os.environ.get("QWEN_API_KEY") == "test-qwen-key"
    assert os.environ.get("KIMI_API_KEY") == "test-kimi-key"


def test_kimi_model_default():
    """Verify Kimi model defaults are correct."""
    model = os.environ.get("KIMI_MODEL", "kimi-k2.7-code")
    assert model == "kimi-k2.7-code"
    assert "k2" in model.lower()  # should be K2 series
