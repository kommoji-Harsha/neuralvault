"""Unit tests verifying Docker configuration and offline documentation presence."""

from pathlib import Path


def test_docker_files_exist():
    dockerfile = Path("Dockerfile")
    docker_compose = Path("docker-compose.yml")
    offline_doc = Path("docs/offline.md")

    assert dockerfile.exists()
    assert docker_compose.exists()
    assert offline_doc.exists()

    df_content = dockerfile.read_text(encoding="utf-8")
    assert "8042" in df_content

    dc_content = docker_compose.read_text(encoding="utf-8")
    assert "neuralvault-api" in dc_content
    assert "neuralvault-mcp" in dc_content
