#!/usr/bin/env python3
"""kimi-bridge: Kimi/Moonshot API MCP Bridge v1.0.0

通过 MCP 协议将 Kimi K2 模型能力暴露给 Claude Code。
专注场景：深度合成、长文本推理、视频理解（Kimi 原生支持）。

Architecture:
  Claude Code (主模型) → MCP stdio → kimi-bridge → Kimi API (api.moonshot.cn)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kimi-bridge")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")
KIMI_BASE_URL = os.environ.get(
    "KIMI_BASE_URL",
    "https://api.moonshot.cn/v1",
)
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-k2.7-code")  # Kimi K2.7 Code 编程模型
REQUEST_TIMEOUT = 180  # Kimi 思考可能较慢

READ_ONLY_HINT = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

# ---------------------------------------------------------------------------
# Kimi API 客户端
# ---------------------------------------------------------------------------


class KimiClient:
    """Kimi/Moonshot API 客户端（OpenAI-compatible）。"""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def ready(self) -> bool:
        return bool(KIMI_API_KEY)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized — call init() first")
        return self._client

    async def init(self):
        self._client = httpx.AsyncClient(
            base_url=KIMI_BASE_URL.rstrip("/"),
            headers={
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """调用 Kimi Chat Completions API。"""
        data = {
            "model": model or KIMI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self._post("/chat/completions", data)
        return resp["choices"][0]["message"]["content"]

    async def _post(self, path: str, json_data: dict, retries: int = 1) -> dict:
        """HTTP POST 调用，带自动重试。"""
        for attempt in range(retries + 1):
            try:
                resp = await self.client.post(path, json=json_data)
                if resp.status_code == 200:
                    return resp.json()

                body = resp.text[:500]
                if attempt < retries and 500 <= resp.status_code < 600:
                    logger.warning(
                        "Kimi API %d (attempt %d/%d), retrying...",
                        resp.status_code, attempt + 1, retries,
                    )
                    await asyncio.sleep(1.5 ** attempt)
                    continue

                raise RuntimeError(
                    f"Kimi API error {resp.status_code}: {body}"
                )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < retries:
                    logger.warning(
                        "Kimi connection error (attempt %d/%d), retrying...",
                        attempt + 1, retries,
                    )
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                raise RuntimeError(
                    f"Kimi API connection error after {retries + 1} attempts: {e}"
                ) from e

        raise RuntimeError(f"Unexpected error calling Kimi API")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("kimi-bridge", version="1.0.0")
kimi = KimiClient()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="kimi_chat",
            description=(
                "调用 Kimi K2 模型进行深度推理和综合。"
                "适用场景：多模态综合、长文本分析、复杂推理、视频内容理解（Kimi 原生支持视频帧提取）。"
                "注意：Kimi K2 是思考模型，响应可能较慢（30-120s），请耐心等待。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "发送给 Kimi 的提示词/问题",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "系统提示词（可选），定义 Kimi 的角色和行为",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "温度参数，0.0-1.0，默认 0.7。分析类任务建议 0.3",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "最大输出 tokens，默认 4096",
                        "minimum": 1,
                        "maximum": 16384,
                    },
                },
                "required": ["prompt"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="kimi_synthesize",
            description=(
                "Kimi 多源综合工具。将多段分析结果（如多张图的 vision_analyze 输出）"
                "发送给 Kimi 进行深度综合和关联分析。适合 L2 级别的跨模态合成。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "description": "多源分析结果列表，每项为 {label: 来源标签, content: 分析内容}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "来源标签，如 '图1：销售额趋势'",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "该来源的分析内容/结构化 JSON",
                                },
                            },
                            "required": ["label", "content"],
                        },
                    },
                    "question": {
                        "type": "string",
                        "description": "要回答的综合问题，如「这三张图共同说明了什么趋势？」",
                    },
                },
                "required": ["sources", "question"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="kimi_health",
            description="检查 Kimi API 的可用状态。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            annotations=READ_ONLY_HINT,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    try:
        if name == "kimi_chat":
            return await _handle_kimi_chat(arguments)
        elif name == "kimi_synthesize":
            return await _handle_kimi_synthesize(arguments)
        elif name == "kimi_health":
            return await _handle_kimi_health()
        else:
            return error_result(f"未知工具: {name}")
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return error_result(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


async def _handle_kimi_chat(args: dict) -> CallToolResult:
    prompt = args.get("prompt", "")
    system_prompt = args.get("system_prompt")
    temperature = args.get("temperature", 0.7)
    max_tokens = args.get("max_tokens", 4096)

    if not prompt:
        return error_result("prompt 不能为空")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = await kimi.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return ok_result(result)


async def _handle_kimi_synthesize(args: dict) -> CallToolResult:
    sources = args.get("sources", [])
    question = args.get("question", "")

    if not sources:
        return error_result("sources 不能为空")
    if not question:
        return error_result("question 不能为空")

    # 拼接多源内容为结构化提示
    parts = [f"## 综合任务\n{question}\n\n## 多源分析结果\n"]
    for i, src in enumerate(sources, 1):
        label = src.get("label", f"来源{i}")
        content = src.get("content", "")
        parts.append(f"### {label}\n{content}\n")

    parts.append(
        "\n## 请综合以上所有来源，回答综合任务中的问题。\n"
        "要求：发现跨源关联、冲突和深层模式，给出有依据的综合结论。"
    )

    system_prompt = (
        "你是一位多模态合成专家。你的任务是对多个分析来源进行综合，"
        "发现跨源关联、不一致和深层模式。输出结构化、有依据的综合报告。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]

    result = await kimi.chat(messages, temperature=0.3)
    return ok_result(result)


async def _handle_kimi_health() -> CallToolResult:
    statuses = {
        "kimi-k2.7-code": {
            "ready": kimi.ready,
            "capabilities": [
                "deep_reasoning",
                "multimodal_synthesis",
                "long_context",
                "video_understanding",
            ],
            "model": KIMI_MODEL,
            "base_url": KIMI_BASE_URL,
            "note": (
                "Kimi K2 思考模型 — L2 综合能力就绪"
                if kimi.ready
                else "KIMI_API_KEY 未设置"
            ),
        },
    }
    return ok_result(json.dumps(statuses, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def ok_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


def error_result(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run():
    if kimi.ready:
        await kimi.init()
        logger.info("Kimi K2 适配器就绪 (model=%s)", KIMI_MODEL)
    else:
        logger.warning("KIMI_API_KEY 未设置 — Kimi bridge 不可用")

    logger.info("kimi-bridge 启动完毕，等待 Claude Code 连接...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    if kimi.ready:
        await kimi.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
