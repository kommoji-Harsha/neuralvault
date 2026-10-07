.PHONY: install lint test clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
