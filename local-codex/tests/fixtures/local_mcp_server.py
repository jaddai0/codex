#!/usr/bin/env python3
"""Tiny stdio MCP server used by the Local Codex acceptance test."""

from __future__ import annotations

import json
import sys


def respond(request_id: object, result: object) -> None:
    print(
        json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True
    )


for line in sys.stdin:
    request = json.loads(line)
    request_id = request.get("id")
    method = request.get("method")
    if method == "initialize":
        respond(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "local-codex-canary", "version": "1"},
            },
        )
    elif method == "tools/list":
        respond(
            request_id,
            {
                "tools": [
                    {
                        "name": "local_echo",
                        "description": "Return a local MCP acceptance marker.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    }
                ]
            },
        )
    elif method == "tools/call":
        text = request.get("params", {}).get("arguments", {}).get("text", "")
        respond(
            request_id,
            {
                "content": [{"type": "text", "text": f"MCP_LOCAL_OK:{text}"}],
                "isError": False,
            },
        )
    elif request_id is not None:
        respond(request_id, {})
