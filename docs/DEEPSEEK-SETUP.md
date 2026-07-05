# 将 Claude Code 接入 DeepSeek

cn-llm-bridge 解决的是多模态扩展问题，但 Claude Code 本身的推理也需要一个模型来驱动。本教程介绍如何让 Claude Code 使用 DeepSeek 作为主推理模型。

## 为什么用 DeepSeek？

Claude Code 本质上是一个 CLI 调度器——它发起请求、读取响应、调用工具。驱动它的模型不一定是 Claude 官方模型。DeepSeek 的优势：

- **性价比极高**：V3 的代码和推理能力媲美顶级模型，价格仅为其 1/10
- **OpenAI 兼容 API**：无需额外适配层，直接配置
- **上下文窗口大**：128K token，适合长任务

## 三种接入方式

### 方式一：直接配置 API Base URL（推荐）

在 Claude Code 的配置文件 `~/.claude/settings.json` 中设置：

```json
{
  "apiKeyHelper": "echo $DEEPSEEK_API_KEY",
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic/v1"
  }
}
```

> DeepSeek 提供了 Anthropic-compatible API 端点，Claude Code 可以直接对接。

然后设置环境变量：

```bash
# ~/.bashrc 或 ~/.zshrc
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
```

获取 Key：[platform.deepseek.com](https://platform.deepseek.com/)

### 方式二：通过中转站（更稳定）

如果你需要更稳定的连接，可以使用 API 中转服务：

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "sk-your-key",
    "ANTHROPIC_BASE_URL": "https://your-proxy.com/v1"
  }
}
```

### 方式三：LiteLLM 本地代理

用 LiteLLM 在本地启动一个 OpenAI 兼容代理，将任意模型包装成 Claude Code 能用的格式：

```bash
pip install litellm
```

创建 `litellm_config.yaml`：

```yaml
model_list:
  - model_name: deepseek-chat
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: sk-your-deepseek-key
  - model_name: deepseek-reasoner
    litellm_params:
      model: deepseek/deepseek-reasoner
      api_key: sk-your-deepseek-key
```

启动代理：

```bash
litellm --config litellm_config.yaml --port 4000
```

然后配置 Claude Code 指向本地代理：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000/v1"
  }
}
```

## 完整工作流

配好之后，你的完整 AI 工作台架构就是：

```
你的指令
    │
    ▼
Claude Code（调度层）
    │ 主推理: DeepSeek V3 / R1
    │ 审查: Claude（通过 API Key）
    │
    ├── 文本推理 ──→ DeepSeek API
    │
    ├── 图片分析 ──→ cn-llm-bridge ──→ Qwen 视觉
    │
    ├── 音频转写 ──→ cn-llm-bridge ──→ qwen3-asr-flash
    │                                 └→ faster-whisper（兜底）
    │
    └── 深度合成 ──→ kimi-bridge ──→ Kimi K2
```

## 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 日常编码、问答 | DeepSeek V3 | 快、便宜、代码能力强 |
| 复杂推理、规划 | DeepSeek R1 | 深度思考链，逻辑严密 |
| 代码审查、最终把关 | Claude（官方 API） | 作为独立 Evaluator，盲点不重叠 |
| 图片/音频 | cn-llm-bridge 子模型 | 专模型专任务，避免主模型分心 |

## 常见问题

**Q: 用 DeepSeek 代替 Claude 官模，效果会不会差很多？**

A: 日常编码和推理任务差距不大。真正拉开差距的是"Generator + Evaluator 双模型"架构——DeepSeek 生成 + Claude 审查，质量反而比单一 Claude 更好（见 README 原则一）。

**Q: DeepSeek 的 Anthropic-compatible API 稳定吗？**

A: DeepSeek 官方维护的适配端点，兼容 Messages API 和 Tool Use。如果遇到兼容性问题，用 LiteLLM 代理方案最稳定。

**Q: 能完全不用 Claude 官方 API 吗？**

A: 可以，但不推荐。Claude Code 的调度逻辑依赖 Claude 模型的理解能力，完全替换可能导致工具调用的准确率下降。建议保留 Claude 作为最小化调度核心，把大量推理工作分流给 DeepSeek。

## 相关资源

- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [LiteLLM 文档](https://docs.litellm.ai/)
- [Claude Code 配置指南](https://docs.anthropic.com/en/docs/claude-code/settings)
