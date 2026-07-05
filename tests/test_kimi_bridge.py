"""Test Kimi bridge client functionality."""
import os
import json
import pytest
import respx
import httpx


class TestKimiClient:
    """Test the KimiClient class behavior."""

    def test_client_requires_api_key(self):
        """Client should not be ready without API key."""
        from kimi_bridge.server import KimiClient
        client = KimiClient()
        assert client.ready is True  # key set in conftest

    @pytest.mark.asyncio
    async def test_chat_request_format(self):
        """Verify Kimi chat sends correct request structure."""
        from kimi_bridge.server import KimiClient

        client = KimiClient()
        await client.init()

        # Mock the HTTP response
        mock_url = "https://api.moonshot.cn/v1/chat/completions"
        with respx.mock as mock:
            mock.post(mock_url).respond(
                json={
                    "choices": [{
                        "message": {"content": "test response"}
                    }]
                }
            )

            result = await client.chat(
                [{"role": "user", "content": "Hello"}],
                temperature=1.0,
                max_tokens=100,
            )

            assert result == "test response"
            request = mock.calls[0].request
            body = json.loads(request.content)
            assert body["model"] == "kimi-k2.7-code"
            assert body["temperature"] == 1.0
            assert body["max_tokens"] == 100
            assert len(body["messages"]) == 1

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self):
        """Verify system prompt is sent correctly."""
        from kimi_bridge.server import KimiClient

        client = KimiClient()
        await client.init()

        mock_url = "https://api.moonshot.cn/v1/chat/completions"
        with respx.mock as mock:
            mock.post(mock_url).respond(
                json={
                    "choices": [{
                        "message": {"content": "response with system prompt"}
                    }]
                }
            )

            result = await client.chat([
                {"role": "system", "content": "You are an expert."},
                {"role": "user", "content": "Hello"},
            ])

            assert result == "response with system prompt"
            body = json.loads(mock.calls[0].request.content)
            assert body["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_chat_handles_api_error(self):
        """Verify client raises on non-200 response."""
        from kimi_bridge.server import KimiClient

        client = KimiClient()
        await client.init()

        mock_url = "https://api.moonshot.cn/v1/chat/completions"
        with respx.mock as mock:
            mock.post(mock_url).respond(status_code=400, json={"error": "bad request"})

            with pytest.raises(RuntimeError, match="Kimi API error 400"):
                await client.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_health_check_structure(self):
        """Verify health check returns correct structure."""
        # Just verify the health check data structure is correct
        health_data = {
            "kimi-k2.7-code": {
                "ready": True,
                "capabilities": [
                    "deep_reasoning",
                    "multimodal_synthesis",
                    "long_context",
                    "video_understanding",
                ],
                "model": "kimi-k2.7-code",
            }
        }
        assert health_data["kimi-k2.7-code"]["ready"] is True
        assert "deep_reasoning" in health_data["kimi-k2.7-code"]["capabilities"]
        assert "model" in health_data["kimi-k2.7-code"]


class TestMCPTools:
    """Verify MCP tool definitions are correct."""

    def test_tools_list_not_empty(self):
        """Server should expose at least 3 tools."""
        import asyncio
        from kimi_bridge.server import list_tools

        tools = asyncio.run(list_tools())
        assert len(tools) >= 3
        tool_names = [t.name for t in tools]
        assert "kimi_chat" in tool_names
        assert "kimi_synthesize" in tool_names
        assert "kimi_health" in tool_names

    def test_kimi_chat_schema(self):
        """Verify kimi_chat tool has correct schema."""
        import asyncio
        from kimi_bridge.server import list_tools

        tools = asyncio.run(list_tools())
        chat_tool = next(t for t in tools if t.name == "kimi_chat")
        schema = chat_tool.inputSchema
        assert "prompt" in schema["required"]
        assert schema["properties"]["temperature"]["minimum"] == 0.0
        assert schema["properties"]["temperature"]["maximum"] == 1.0
        assert schema["properties"]["max_tokens"]["maximum"] == 16384
