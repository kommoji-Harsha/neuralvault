FROM python:3.12-slim

WORKDIR /app

# Install build tools and python dependencies
COPY pyproject.toml README.md collections.yaml ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

EXPOSE 8042

CMD ["uvicorn", "neuralvault.api.app:app", "--host", "0.0.0.0", "--port", "8042"]
