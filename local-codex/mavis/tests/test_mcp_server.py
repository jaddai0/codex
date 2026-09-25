"""The installed memory MCP uses the launcher-bound project, not agent paths."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mavis.project_evidence import project_home
from mavis.mcp_server import bound_homes
from mavis.transcripts import TranscriptArchive


class MemoryMcpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mavis-memory-mcp-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        subprocess.run(["git", "init", "-q", str(self.project)], check=True)
        self.service = self.root / "service"
        self.service.mkdir()
        archive = TranscriptArchive(project_home(self.project, create=True), "design")
        archive.append_segment([{"role": "user", "content":
                                 "Accepted checkout rule: shipping is not taxed."}])

    async def call_server(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mavis.mcp_server"],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                 "MAVIS_HOME": str(self.service),
                 "MAVIS_PROJECT_ROOT": str(self.project)},
        )
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.assertEqual([tool.name for tool in tools.tools],
                                 ["mavis_librarian_ask", "mavis_jev_advise"])
                self.assertNotIn("project", tools.tools[0].inputSchema["properties"])
                jev_inputs = tools.tools[1].inputSchema["properties"]
                self.assertEqual(set(jev_inputs), {"purpose", "signals", "estimated_cost_usd"})
                result = await session.call_tool("mavis_librarian_ask", {
                    "conversation_id": "design",
                    "question": "What was accepted for shipping?",
                    "search_terms": ["checkout"],
                    "no_model": True,
                })
                self.assertFalse(result.isError)
                payload = result.structuredContent
                if payload is None:
                    payload = json.loads(result.content[0].text)
                self.assertFalse(payload["model_called"])
                self.assertIn("shipping is not taxed", payload["answer"])
                self.assertEqual(len(payload["citations"]), 1)

    def test_stdio_no_model_archive_citation(self):
        asyncio.run(self.call_server())

    def test_unbound_project_is_rejected(self):
        with patch.dict(os.environ, {"MAVIS_HOME": str(self.service)}, clear=True):
            with self.assertRaisesRegex(ValueError, "MAVIS_PROJECT_ROOT"):
                bound_homes()


if __name__ == "__main__":
    unittest.main()
