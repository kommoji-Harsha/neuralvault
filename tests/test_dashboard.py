"""Unit tests for dashboard and neutrality configurations."""

from neuralvault.client.client import RagClient
from neuralvault.config import load_collections_config
from neuralvault.dashboard.app import load_benchmark_results


def test_neutrality_collections_config():
    cols = load_collections_config()
    assert "my-project" in cols
    assert "assistant-project" not in cols
    assert "Any project" in cols["my-project"]["description"]


def test_rag_client_neutral_defaults():
    client = RagClient(mode="local")
    assert client.default_collection == "my-project"


def test_dashboard_benchmark_loader():
    data = load_benchmark_results("results/results.json")
    assert "collection" in data or data == {}
