#!/usr/bin/env python3
"""cn-llm-bridge: 中国大模型 MCP Bridge v1.0.0

通过 MCP 协议将中国大模型能力暴露给 Claude Code。
Claude Code 保持主模型不变，通过此桥接调用各模型的专长能力。

Architecture:
  Claude Code (主模型) → MCP stdio → cn-llm-bridge → 各中国大模型 API
                                    ├── Qwen3.7-Plus  (阿里百炼) → 视觉
                                    ├── Qwen3.5-Omni  (阿里百炼) → 音频  (Phase 2)
                                    ├── MiniMax M3                → Agent (Phase 3)
                                    └── GLM-5.1       (智谱AI)   → 长周期 (Phase 3)
"""

import asyncio
import json
import logging
import os
import struct
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

# faster-whisper — 本地音频转写（CPU-only，不依赖 GPU）
_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel

    _WHISPER_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cn-llm-bridge")

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

# 从环境变量读取 — API Key 不硬编码
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
BAILIAN_BASE_URL = os.environ.get(
    "BAILIAN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 模型 ID
QWEN_VISION_MODEL = "qwen3.7-plus"
QWEN_AUDIO_MODEL = "qwen3-asr-flash"

# 超时
REQUEST_TIMEOUT = 120  # 秒

# 工具注解
READ_ONLY_HINT = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

# faster-whisper 配置（CPU-only，已验证 small + INT8 在 Windows 上可行）
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")  # tiny/base/small/medium/large-v3
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")  # cpu 或 cuda
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # int8/float16/int8_float16
WHISPER_MODEL_DIR = os.environ.get(
    "WHISPER_MODEL_DIR",
    str(Path.home() / ".cache" / "faster-whisper-models"),
)

# 离线模式：不从 HF Hub 下载，全部使用本地缓存
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# AudioTranscriber — 本地 faster-whisper 实例（惰性加载）
_whisper_model: Any = None  # WhisperModel 实例（线程安全）

# Project 根目录 (用于任务持久化)
PROJECT_ROOT = Path(__file__).resolve().parent
TASKS_FILE = PROJECT_ROOT / "tasks.json"

# ---------------------------------------------------------------------------
# ModelAdapter — 抽象基类
# ---------------------------------------------------------------------------


class ModelAdapter(ABC):
    """模型适配器基类。每个模型实现一个子类。"""

    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized — call init() first")
        return self._client

    async def init(self):
        """初始化 HTTP 客户端。在 server 启动时调用。"""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """通用文本/多模态对话。"""
        ...

    async def _post(self, path: str, json_data: dict, retries: int = 1) -> dict:
        """OpenAI-compatible API 调用，带自动重试。

        对 5xx 和服务端连接错误自动重试，指数退避。
        """
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = await self.client.post(path, json=json_data)
                if resp.status_code == 200:
                    return resp.json()

                body = resp.text[:500]
                # 5xx 可重试
                if attempt < retries and 500 <= resp.status_code < 600:
                    logger.warning(
                        "API %d from %s (attempt %d/%d), retrying...",
                        resp.status_code,
                        self.model_id,
                        attempt + 1,
                        retries,
                    )
                    await asyncio.sleep(1.5**attempt)
                    continue

                raise RuntimeError(f"API error {resp.status_code} from {self.model_id}: {body}")

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        "Connection error to %s (attempt %d/%d), retrying...",
                        self.model_id,
                        attempt + 1,
                        retries,
                    )
                    await asyncio.sleep(1.5**attempt)
                    continue
                raise RuntimeError(
                    f"API connection error from {self.model_id} after {retries + 1} attempts: {e}"
                ) from e

        # 理论上不会到这里，但兜底
        raise RuntimeError(f"Unexpected error calling {self.model_id}") from last_error

    def health(self) -> bool:
        """检查此适配器是否可工作（API Key 是否存在）。"""
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# QwenVisionAdapter — Qwen3.7-Plus 视觉/文本
# ---------------------------------------------------------------------------


class QwenVisionAdapter(ModelAdapter):
    """Qwen3.7-Plus 适配器：图像理解 + 文本对话。"""

    def __init__(self):
        super().__init__(QWEN_API_KEY, BAILIAN_BASE_URL, QWEN_VISION_MODEL)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """通用对话。messages 可包含文本和 image_url 内容块。"""
        data = {
            "model": self.model_id,
            "messages": messages,
            **kwargs,
        }
        result = await self._post("/chat/completions", data)
        return result["choices"][0]["message"]["content"]

    async def vision(
        self,
        prompt: str,
        image_data: str | None = None,
        image_url: str | None = None,
        detail: str = "auto",
    ) -> str:
        """图像理解。

        Args:
            prompt: 对图像的提问或描述指令
            image_data: base64 编码的图像数据（兼容带 data: 前缀）
            image_url: 图像的公开 URL（与 image_data 二选一）
            detail: 细节级别 "auto" | "low" | "high"
        """
        if not image_data and not image_url:
            return "错误：必须提供 image_data 或 image_url"

        content: list[dict] = [{"type": "text", "text": prompt}]

        if image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": detail},
                }
            )
        else:
            raw_data, mime = _parse_image_data(image_data or "")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{raw_data}",
                        "detail": detail,
                    },
                }
            )

        messages = [{"role": "user", "content": content}]
        return await self.chat(messages)


# ---------------------------------------------------------------------------
# QwenOmniAdapter — Qwen3.5-Omni 音频/视频
# ---------------------------------------------------------------------------


class QwenAsrAdapter(ModelAdapter):
    """Qwen-ASR 适配器：云端音频转写。

    使用阿里百炼 qwen3-asr-flash 模型，通过 OpenAI 兼容的 /chat/completions API。
    不依赖本地模型，API Key 即可工作。

    工作流程：
    1. ffmpeg 将任意音频转为 WAV (16kHz mono PCM)
    2. 按 ~4 MB 分块（适应 API 6 MB 请求体限制）
    3. 每块以纯 input_audio 格式发送（不能混 text）
    4. 拼接所有分块转写结果
    """

    # 文件扩展名 → MIME 映射（用于 data URI）
    _MIME_MAP = {
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
        ".webm": "audio/webm",
    }

    # 每个分块的最大 PCM 字节数（base64 编码后约 5.3 MB，加 JSON 约 5.5 MB，安全低于 6 MB 限制）
    _MAX_CHUNK_PCM_BYTES = 4 * 1024 * 1024

    def __init__(self):
        super().__init__(QWEN_API_KEY, BAILIAN_BASE_URL, QWEN_AUDIO_MODEL)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """通用文本对话（ASR 模型也支持纯文本）。"""
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        result = await self._post("/chat/completions", data)
        return result["choices"][0]["message"]["content"]

    def _to_wav(self, file_path: str) -> bytes:
        """用 ffmpeg 将任意音频转为 WAV (16kHz mono PCM)。"""
        import os as _os
        import subprocess
        import tempfile

        wav_path = _os.path.join(
            tempfile.gettempdir(),
            f"cn_llm_bridge_asr_{_os.getpid()}.wav",
        )
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    file_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    wav_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg 转换失败: {result.stderr[:300]}")
            wav_bytes = Path(wav_path).read_bytes()
            return wav_bytes
        finally:
            try:
                _os.unlink(wav_path)
            except OSError:
                pass

    async def _transcribe_chunk(
        self,
        chunk_wav: bytes,
        language: str | None,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> str:
        """转写单个 WAV 块（纯 input_audio，不混 text）。"""
        import base64

        b64 = base64.b64encode(chunk_wav).decode("utf-8")
        data_uri = f"data:audio/wav;base64,{b64}"

        data: dict = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_uri},
                        }
                    ],
                }
            ],
            "asr_options": {"enable_itn": False},
        }
        if language:
            data["asr_options"]["language"] = language

        label = f"第 {chunk_index + 1}/{total_chunks} 块" if total_chunks > 1 else ""
        logger.info("云端转写 %s (%.0f KB)", label, len(chunk_wav) / 1024)

        result = await self._post("/chat/completions", data, retries=2)
        return result["choices"][0]["message"]["content"].strip()

    async def transcribe(
        self,
        file_path: str,
        language: str | None = None,
    ) -> dict:
        """云端音频转写（qwen3-asr-flash）。

        流程：原始音频 → ffmpeg 转 WAV → 分块 → input_audio 转写 → 拼接。

        Args:
            file_path: 音频文件的绝对路径（支持 m4a/mp3/wav/ogg/flac 等）
            language: 语言代码（如 "zh"），用于提升识别准确率

        Returns:
            {"text": "...", "segments": [], "language": "zh",
             "duration_s": 0, "provider": "qwen3-asr-flash (cloud)"}
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        if not p.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        file_size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("读取音频文件: %s (%.1f MB)", p.name, file_size_mb)

        # 转为 WAV (16kHz mono PCM)
        wav_bytes = self._to_wav(file_path)
        logger.info("转为 WAV: %.1f MB (16kHz mono)", len(wav_bytes) / (1024 * 1024))

        # WAV 头 44 字节，后面是 PCM 数据
        wav_header = wav_bytes[:44]
        pcm_data = wav_bytes[44:]

        # 分块转写
        if len(pcm_data) <= self._MAX_CHUNK_PCM_BYTES:
            text = await self._transcribe_chunk(
                wav_bytes,
                language,
                chunk_index=0,
                total_chunks=1,
            )
        else:
            import math

            num_chunks = math.ceil(len(pcm_data) / self._MAX_CHUNK_PCM_BYTES)
            logger.info(
                "WAV PCM %.1f MB，分 %d 块转写",
                len(pcm_data) / (1024 * 1024),
                num_chunks,
            )

            texts = []
            for i in range(num_chunks):
                start = i * self._MAX_CHUNK_PCM_BYTES
                end = min(start + self._MAX_CHUNK_PCM_BYTES, len(pcm_data))
                chunk_pcm = pcm_data[start:end]

                # 重建有效 WAV 文件头（更新 data size 字段）
                chunk_wav = wav_header[:40] + struct.pack("<I", len(chunk_pcm)) + wav_header[44:] + chunk_pcm

                chunk_text = await self._transcribe_chunk(
                    chunk_wav,
                    language,
                    chunk_index=i,
                    total_chunks=num_chunks,
                )
                texts.append(chunk_text)

            text = "\n".join(texts)

        return {
            "text": text,
            "segments": [],
            "language": language or "auto",
            "duration_s": 0,
            "provider": "qwen3-asr-flash (cloud)",
        }


# ---------------------------------------------------------------------------
# TaskRegistry — Agent 任务状态的文件持久化
# ---------------------------------------------------------------------------


@dataclass
class TaskState:
    id: str
    model: str
    status: str  # "running" | "completed" | "failed"
    created_at: float
    task: str
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None
    progress: str = ""


class TaskRegistry:
    """任务注册表，JSON 文件持久化。

    Phase 3 的 Agent 任务使用此 registry 管理生命周期。
    Phase 1 先搭好架子，Phase 2/3 填充实际逻辑。
    """

    def __init__(self, path: Path = TASKS_FILE):
        self._path = path
        self._tasks: dict[str, TaskState] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for item in data:
                    ts = TaskState(**item)
                    self._tasks[ts.id] = ts
                logger.info("Loaded %d tasks from %s", len(self._tasks), self._path.name)
            except Exception as e:
                logger.warning("Failed to load tasks: %s", e)

    async def _save(self):
        async with self._lock:
            data = [asdict(t) for t in self._tasks.values()]
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def create(self, model: str, task: str) -> str:
        task_id = uuid.uuid4().hex[:12]
        self._tasks[task_id] = TaskState(
            id=task_id,
            model=model,
            status="running",
            created_at=time.time(),
            task=task,
        )
        await self._save()
        return task_id

    async def update(self, task_id: str, **kwargs):
        if task_id in self._tasks:
            for k, v in kwargs.items():
                setattr(self._tasks[task_id], k, v)
            await self._save()

    def get(self, task_id: str) -> TaskState | None:
        return self._tasks.get(task_id)

    def list_active(self) -> list[TaskState]:
        return [t for t in self._tasks.values() if t.status == "running"]


# ---------------------------------------------------------------------------
# AudioTranscriber — faster-whisper 本地转写
# ---------------------------------------------------------------------------


def _get_whisper_model():
    """惰性加载 faster-whisper 模型（线程安全，只加载一次）。"""
    global _whisper_model
    if not _WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        logger.info(
            "加载 faster-whisper 模型: size=%s device=%s compute=%s",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=WHISPER_MODEL_DIR,
            local_files_only=True,
        )
        logger.info("faster-whisper 模型加载完成")
    return _whisper_model


async def _transcribe_audio(
    file_path: str,
    language: str | None = None,
    task: str = "transcribe",
) -> dict:
    """异步音频转写。在 executor 中运行同步的 faster-whisper。

    Args:
        file_path: 音频文件的绝对路径（m4a/mp3/wav/ogg/flac 等）
        language: 语言代码（如 "zh", "en"），None 为自动检测
        task: "transcribe"（原语言输出）或 "translate"（翻译为英文）

    Returns:
        {"segments": [...], "language": "zh", "duration_s": 83.2, "text": "全文..."}
    """
    model = _get_whisper_model()
    if model is None:
        raise RuntimeError("faster-whisper 未安装。请运行: pip install faster-whisper")

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    if not p.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    # faster-whisper 的 transcribe 是同步阻塞的，需要在 executor 中运行
    loop = asyncio.get_running_loop()

    def _run():
        segments_raw, info = model.transcribe(
            str(p),
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,  # 自动过滤静音段
        )
        segments = []
        full_text_parts = []
        for seg in segments_raw:
            segments.append(
                {
                    "id": seg.id,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                }
            )
            full_text_parts.append(seg.text.strip())
        return {
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_s": round(info.duration, 1),
            "text": "".join(full_text_parts),
        }

    return await loop.run_in_executor(None, _run)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("cn-llm-bridge", version="1.0.0")

# 全局适配器实例
qwen_vision = QwenVisionAdapter()
qwen_asr = QwenAsrAdapter()
task_registry = TaskRegistry()


@server.list_tools()
async def list_tools() -> list[Tool]:
    tools = [
        Tool(
            name="vision_analyze",
            description="分析图像内容。支持 URL 或 base64 编码的图像。返回结构化 JSON（含摘要、文字、物体列表）。适合一次性图像分析。",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "对图像的提问或描述指令，如「描述这张图」",
                    },
                    "image_url": {
                        "type": "string",
                        "description": "图像的公开 URL（与 image_data 二选一）",
                    },
                    "image_data": {
                        "type": "string",
                        "description": "base64 编码的图像数据（兼容带 data:image/xxx;base64, 前缀，与 image_url 二选一）",
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["auto", "low", "high"],
                        "description": "细节级别，默认 auto（会根据 prompt 自动推断）",
                    },
                },
                "required": ["prompt"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="vision_chat",
            description="多模态对话。支持含图片的多轮对话，自动简洁回复。如果需要一次性结构化分析（JSON 格式输出），请用 vision_analyze。",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "对话消息列表。content 支持纯文本字符串或图文数组",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"],
                                },
                                "content": {
                                    "description": '文本字符串，或图文内容块数组 [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "...", "detail": "auto"}}]',
                                },
                            },
                        },
                    },
                },
                "required": ["messages"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="audio_transcribe",
            description="本地音频转写。使用 faster-whisper small 模型（CPU-only，INT8 量化）将音频文件转为文字。支持 m4a/mp3/wav/ogg/flac 等常见格式。返回分段文本和全文。",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "音频文件的绝对路径，如 D:\\recordings\\interview.m4a",
                    },
                    "language": {
                        "type": "string",
                        "description": "语言代码（如 'zh' 中文, 'en' 英文），不指定则自动检测",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["transcribe", "translate"],
                        "description": "transcribe（原语言输出，默认）或 translate（翻译为英文）",
                    },
                },
                "required": ["file_path"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="tools_health",
            description="检查所有模型 API 的可用状态。返回各模型是否就绪。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            annotations=READ_ONLY_HINT,
        ),
    ]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    try:
        if name == "vision_analyze":
            return await _handle_vision_analyze(arguments)
        elif name == "vision_chat":
            return await _handle_vision_chat(arguments)
        elif name == "audio_transcribe":
            return await _handle_audio_transcribe(arguments)
        elif name == "tools_health":
            return await _handle_tools_health()
        else:
            return error_result(f"未知工具: {name}")
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return error_result(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


async def _handle_vision_analyze(args: dict) -> CallToolResult:
    prompt = args.get("prompt", "")
    image_url = args.get("image_url")
    image_data = args.get("image_data")
    detail = _infer_detail_level(prompt, args.get("detail", "auto"))

    if not prompt:
        return error_result("prompt 不能为空")
    if not image_url and not image_data:
        return error_result("必须提供 image_url 或 image_data")

    # 构建消息：system 指令（含复杂度自评估 + 字数自约束）
    system_prompt = (
        "你必须以 JSON 格式回复，包含以下字段：\n"
        '- complexity: 你评估的任务复杂度（"simple" / "medium" / "complex"），'
        "simple 表示简单是非/有无问题，medium 表示一般描述，complex 表示需要深入分析\n"
        "- summary: 一句话摘要（5-15 字）\n"
        "- details: 详细描述\n"
        "- text_found: 识别到的文字（如无文字则为 null）\n"
        "- objects: 检测到的关键物体/元素列表（数组，可为空）\n"
        "根据你判断的 complexity 控制 details 长度：\n"
        "- simple: 不超过 30 字\n"
        "- medium: 不超过 100 字\n"
        "- complex: 不超过 300 字\n"
        "只输出 JSON，不要加 markdown 代码块标记。"
    )

    content: list[dict] = [{"type": "text", "text": prompt}]

    if image_url:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},
            }
        )
    else:
        image_data_clean, mime = _parse_image_data(image_data or "")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{image_data_clean}",
                    "detail": detail,
                },
            }
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    max_tokens = _estimate_max_tokens(detail)
    result = await qwen_vision.chat(messages, max_tokens=max_tokens)
    # JSON 校验与修复
    result = _ensure_json(result)
    return ok_result(result)


async def _handle_vision_chat(args: dict) -> CallToolResult:
    messages = args.get("messages", [])
    if not messages:
        return error_result("messages 不能为空")

    # 如果没有 system message，添加默认简洁性约束
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": "请简洁回答，非必要不展开。图片分析控制在 100 字以内。"}] + messages

    # 设一个合理上限，防 runaway 输出
    result = await qwen_vision.chat(messages, max_tokens=1000)
    return ok_result(result)


async def _handle_audio_transcribe(args: dict) -> CallToolResult:
    file_path = args.get("file_path", "")
    language = args.get("language")
    task = args.get("task", "transcribe")

    if not file_path:
        return error_result("file_path 不能为空")

    # ── 主路径：qwen3-asr-flash 云端转写 ──
    if qwen_asr.health():
        try:
            result = await qwen_asr.transcribe(file_path, language=language)
            logger.info("云端转写成功 (qwen3-asr-flash)")
            return ok_result(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("云端转写失败 (%s)，回退到本地 faster-whisper", e)

    # ── 兜底：本地 faster-whisper ──
    try:
        result = await _transcribe_audio(file_path, language=language, task=task)
        return ok_result(json.dumps(result, ensure_ascii=False, indent=2))
    except FileNotFoundError as e:
        return error_result(str(e))
    except RuntimeError as e:
        return error_result(str(e))


async def _handle_tools_health() -> CallToolResult:
    statuses = {
        "qwen3.7-plus": {
            "ready": qwen_vision.health(),
            "capabilities": ["vision", "chat"],
        },
        "qwen3-asr-flash": {
            "ready": qwen_asr.health(),
            "capabilities": ["audio_transcribe"],
            "note": "云端转写（主路径）",
        },
        "faster-whisper": {
            "ready": _WHISPER_AVAILABLE,
            "capabilities": ["audio_transcribe"],
            "model_size": WHISPER_MODEL_SIZE,
            "device": WHISPER_DEVICE,
            "compute": WHISPER_COMPUTE_TYPE,
            "note": "本地 CPU-only 转写" if _WHISPER_AVAILABLE else "faster-whisper 未安装",
        },
        "minimax-m3": {
            "ready": bool(os.environ.get("MINIMAX_API_KEY")),
            "capabilities": ["agent_delegate"],
            "note": "Phase 3 — 尚未实现",
        },
        "glm-5.1": {
            "ready": bool(os.environ.get("GLM_API_KEY")),
            "capabilities": ["agent_delegate"],
            "note": "Phase 3 — 尚未实现",
        },
    }
    return ok_result(json.dumps(statuses, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _estimate_max_tokens(detail: str) -> int:
    """根据 detail 级别预估安全的 max_tokens 上限。

    low → 500（场景描述较短）
    auto → 800（一般用途）
    high → 1500（OCR/精细分析可能需要更多）
    """
    return {"low": 500, "auto": 800, "high": 1500}.get(detail, 800)


def _ensure_json(raw: str) -> str:
    """尝试确保响应是合法 JSON，自动剥离 markdown 代码块包裹。"""
    text = raw.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        # 找到第一个换行后的内容和最后一个 ```
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
        elif "```" in text:
            text = text[: text.rfind("```")].rstrip()
        text = text.strip()

    # 尝试验证 JSON
    try:
        json.loads(text)
        return text  # 合法 JSON，直接返回
    except json.JSONDecodeError:
        # JSON 不合法 — 原样返回（让 caller 决定怎么处理）
        logger.warning("Qwen 返回的 JSON 解析失败，返回原始文本")
        return raw


def ok_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


def error_result(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _parse_image_data(data: str) -> tuple[str, str]:
    """解析 image_data，返回 (纯 base64 数据, MIME 类型)。

    兼容已带 data:image/xxx;base64, 前缀的数据和纯 base64 字符串。
    """
    if data.startswith("data:"):
        # 已有完整 data URI，提取 MIME 和数据
        try:
            header, _, raw = data.partition(",")
            mime = header.replace("data:", "").replace(";base64", "").strip()
            return raw, mime
        except Exception:
            pass
    # 纯 base64 — 从内容前缀猜测 MIME
    mime = _guess_base64_mime(data)
    return data, mime


def _guess_base64_mime(data: str) -> str:
    """从 base64 数据的前几个字符猜测 MIME 类型。"""
    if data.startswith("iVBOR"):
        return "image/png"
    if data.startswith("/9j"):
        return "image/jpeg"
    if data.startswith("R0lGOD"):
        return "image/gif"
    if data.startswith("UklGR"):
        return "image/webp"
    if data.startswith("Qk"):
        return "image/bmp"
    return "image/png"  # 默认


def _infer_detail_level(prompt: str, user_detail: str) -> str:
    """根据 prompt 内容自动推断 Qwen vision 的 detail 级别。

    Args:
        prompt: 用户的提问/指令
        user_detail: 用户显式指定的 detail 值（"auto"/"low"/"high"）
    """
    if user_detail != "auto":
        return user_detail  # 用户显式指定则优先

    ocr_kw = ["读", "文字", "写", "字", "OCR", "提取", "识别", "文本", "内容"]
    scene_kw = ["描述", "场景", "风格", "画面", "整体", "氛围", "概括"]

    for kw in ocr_kw:
        if kw in prompt:
            return "high"
    for kw in scene_kw:
        if kw in prompt:
            return "low"
    return "auto"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run():
    """初始化适配器并启动 MCP server。"""
    qwen_ok = qwen_vision.health()

    if qwen_ok:
        await qwen_vision.init()
        logger.info("Qwen3.7-Plus 适配器就绪")
    else:
        logger.warning("QWEN_API_KEY 未设置 — Qwen3.7-Plus 不可用")

    asr_ok = qwen_asr.health()
    if asr_ok:
        await qwen_asr.init()
        logger.info("Qwen3-ASR-Flash 适配器就绪（云端音频转写）")
    else:
        logger.warning("QWEN_API_KEY 未设置 — Qwen3-ASR-Flash 不可用")

    logger.info("cn-llm-bridge 启动完毕，等待 Claude Code 连接...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    # 关闭
    if qwen_ok:
        await qwen_vision.close()
    if asr_ok:
        await qwen_asr.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
