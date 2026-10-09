# NeuralVault MCP Server Integration Guide

## Overview

NeuralVault runs as a Model Context Protocol (MCP) server exposing four tools:
1. `list_collections`: List available document collections.
2. `search`: Search a collection for relevant passages.
3. `ask`: Perform RAG question answering over a collection with citations.
4. `get_document`: Retrieve document metadata by ID or source file path.

---

## Starting the MCP Server

### Stdio Transport (Claude Desktop / CLI clients)
```bash
neuralvault mcp --transport stdio
```

### HTTP Transport (Streamable SSE / Remote clients)
```bash
neuralvault mcp --transport http --port 8042
```

---

## Client Configurations

### Claude Desktop Configuration
Location:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Config:
```json
{
  "mcpServers": {
    "neuralvault": {
      "command": "neuralvault",
      "args": ["mcp", "--transport", "stdio"]
    }
  }
}
```

### Cursor / Custom MCP Client Configuration
```json
{
  "mcpServers": {
    "neuralvault": {
      "url": "http://localhost:8042/sse",
      "transport": "sse"
    }
  }
}
```
