
# Local OpenAI-Compatible LLM Server with Tool Calling

> 基于 FastAPI + llama-cpp-python 的本地 GGUF 大模型服务，兼容 OpenAI Chat Completions / Responses API，支持 Function Calling，并内置多种工具调用格式解析。

---

## 项目简介

本项目是一个单文件 Python 服务，通过 `llama-cpp-python` 加载本地 GGUF 模型，对外暴露 OpenAI 兼容接口。它适合在本地或内网部署大模型，并为支持 OpenAI API 的客户端、Agent 框架、IDE 插件、Trae 等提供模型服务。

核心目标：

- 兼容 OpenAI `chat/completions` 与 `responses` 接口
- 支持本地 GGUF 模型推理
- 支持工具调用 / Function Calling
- 兼容 Qwen3.8 GGUF `chat_template` 的 XML 风格工具调用
- 兼容旧 Qwen 工具标记与裸 JSON 工具调用
- 可选 MongoDB 请求与对话日志
- 可选 Gradio 界面占位

---

## 核心特性

- **OpenAI 兼容接口**
  - `POST /v1/chat/completions`
  - `POST /v1/responses`
  - `GET /v1/models`
  - `GET /models`
  - `GET /health`

- **本地 GGUF 模型推理**
  - 基于 `llama-cpp-python`
  - 支持自动检测 GGUF 元数据中的 `chat_template`
  - 支持 GPU 层卸载、Flash Attention、KV Cache 等参数调整

- **工具调用 Function Calling**
  - 支持 OpenAI Chat Completions 风格 `tools`
  - 支持 `tool_choice`
  - 支持流式请求下的工具调用返回
  - 支持 Responses API 的 `function_call` 输出

- **多格式工具调用解析**
  - Qwen3.8 GGUF XML 风格：
    ```xml
    <tool_call>
    <function=get_weather>
    <parameter=city>北京</parameter>
    </function>
    </tool_call>
    ```
  - 旧 Qwen 标记：
    ```text
    <|tool_calls_begin|> ... <|tool_calls_end|>
    ```
  - 裸 JSON 工具调用
  - 括号平衡 JSON 提取，修复嵌套参数丢失问题

- **内容规范化与 think 剥离**
  - 支持多模态 `content` 转纯文本
  - 自动剥离 `<think>`、`<|thinking_section|>` 等思考标签

- **工具预筛选**
  - 当工具数量超过 12 个时，根据用户意图筛选相关工具
  - 核心工具永远保留
  - 支持工具组与中英文关键词匹配

- **线程池推理**
  - 同步推理放入线程池，避免阻塞 FastAPI 事件循环
  - 当前推理线程池为单 worker，适合本地单模型场景

- **可选 MongoDB 日志**
  - 记录请求历史与对话历史
  - 未安装 `pymongo` 或连接失败时不影响主服务

- **Gradio 占位界面**
  - 默认未挂载
  - 可通过取消注释挂载到 `/gradio`

---

## 适用场景

- 本地部署 GGUF 模型，并为 OpenAI SDK 客户端提供兼容服务
- 为 Agent / 工作流 / IDE 插件提供本地工具调用模型
- 测试 Qwen3.8 GGUF 的 XML 工具调用模板
- 内网低并发模型服务
- 需要记录请求与对话日志的实验环境

---

## 支持的工具调用格式

服务仅在客户端传入 `tools` 时启用工具调用解析。否则，即使模型模仿输出了工具标记，也会按纯文本返回，避免把代码示例误判为工具调用。

解析优先级：

1. Qwen3.8 GGUF XML 风格工具调用
2. 旧 Qwen `<|tool_calls_begin|>` 标记
3. 裸 JSON 工具调用

### 1. Qwen3.8 GGUF XML 风格

模型输出：

```xml
<tool_call>
<function=get_weather>
<parameter=city>
北京
</parameter>
<parameter=unit>
celsius
</parameter>
</function>
</tool_call>
```

解析后返回：

```json
{
  "id": "call_xxxxxxxx",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
  }
}
```

参数值会先尝试 `json.loads`，能解析成数字、布尔、对象、数组就转换；否则保留原始字符串。

### 2. 旧 Qwen 标记

```text
<|tool_calls_begin|>
<|tool_call_begin|>get_weather<|tool_sep|>{"city": "北京"}<|tool_call_end|>
<|tool_calls_end|>
```

### 3. 裸 JSON

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "北京"
  }
}
```

或：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": {
      "city": "北京"
    }
  }
}
```

---

## 工作流程

1. 接收 OpenAI 兼容请求
2. 将多模态 `content` 规范化为纯文本
3. 解析 `tools` 与 `tool_choice`
4. 如果工具数量超过 12，按用户意图进行工具预筛选
5. 清洗 assistant 消息中的 `tool_calls[].function.arguments`，确保适配 Qwen3.8 `chat_template`
6. 调用 `llama-cpp-python` 进行推理
7. 后处理模型输出：
   - 剥离 think 标签
   - 解析 XML / 旧 Qwen / 裸 JSON 工具调用
   - 标准化为 OpenAI `tool_calls`
8. 返回 Chat Completions 或 Responses API 格式

---

## 项目结构

当前代码为单文件服务，通常可保存为：

```text
.
├── main.py          # 主服务文件，即你提供的代码
├── README.md        # 项目说明
└── ...
```

核心模块均在同一文件中：

- 配置：`Config`
- 日志：`logging`
- 线程池：`_inference_executor`
- MongoDB：`get_mongo_client` / `save_to_mongo`
- 文本与 JSON 工具：`_strip_think`、`_extract_balanced_json`、`_normalize_content`
- 工具调用解析：`parse_tool_calls_from_content`、`parse_xml_tool_calls_from_content`
- 工具预筛选：`_select_relevant_tools`
- 模型管理：`ModelManager`
- OpenAI 应用：`create_openai_app`
- Gradio 占位：`create_gradio_app`
- 主入口：`main`

---

## 环境要求

- Python 3.10+
- `llama-cpp-python`
- `fastapi`
- `uvicorn`
- `pydantic`
- `gradio`
- `torch`
- 可选：`pymongo`
- 本地 GGUF 模型文件

如果需要 GPU 推理，请安装支持 CUDA 的 `llama-cpp-python`。

---

## 安装

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows：

```powershell
.venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -U pip
pip install fastapi uvicorn pydantic llama-cpp-python gradio torch
```

可选 MongoDB：

```bash
pip install pymongo
```

### 3. 安装支持 CUDA 的 llama-cpp-python

Linux / macOS：

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

Windows PowerShell：

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir
```

请根据你的 CUDA 版本、编译环境与 `llama-cpp-python` 官方文档调整。

---

## 配置

### 环境变量

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `MODEL_DIR` | `E:/n_c_r_release/.cache/.gguf` | GGUF 模型目录 |
| `HOST` | `127.0.0.1` | 服务监听地址 |
| `PORT` | `8102` | 服务监听端口 |
| `CHAT_FORMAT` | `None` | 聊天模板格式；为空时从 GGUF 元数据自动检测 |
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB 连接地址 |
| `DB_NAME` | `my_code_db` | MongoDB 数据库名 |

### 代码内配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `MODEL_NAME` | `qwen3.8/Qwen3.8-27B-UD-IQ2_M` | 模型名称，决定模型文件名 |
| `MODEL_PATH` | `${MODEL_DIR}/${MODEL_NAME}.gguf` | 模型完整路径 |
| `MAX_CONTEXT` | `32048 * 2` | 上下文长度，即 `64096` |
| `MAX_TOKENS_DEFAULT` | `4096` | 默认最大生成 token 数 |
| `TOP_P` | `0.8` | 默认 top_p |
| `TEMPERATURE` | `0.6` | 默认 temperature |
| `COLLECTION_CHAT` | `chat_history_gguf` | 对话历史集合 |
| `COLLECTION_REQUEST` | `request_history_gguf` | 请求历史集合 |

模型路径示例：

```text
E:/n_c_r_release/.cache/.gguf/qwen3.8/Qwen3.8-27B-UD-IQ2_M.gguf
```

可通过修改 `Config.MODEL_NAME` 或设置 `MODEL_DIR` 环境变量切换模型。

---

## 启动服务

将代码保存为 `main.py` 后运行：

```bash
python main.py
```

默认服务地址：

```text
http://127.0.0.1:8102
```

健康检查：

```bash
curl http://127.0.0.1:8102/health
```

返回示例：

```json
{
  "status": "ok",
  "model_loaded": true,
  "chat_format": "qwen"
}
```

---

## API 说明

### 路由总览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions 兼容接口 |
| `POST` | `/v1/responses` | OpenAI Responses API 兼容接口 |
| `GET` | `/v1/models` | 模型列表 |
| `GET` | `/models` | 模型列表短路由 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/completions` | 不支持，返回 404 |
| 其他 | `/{path:path}` | 404 兜底 |

---

## Chat Completions

### 非流式文本请求

```bash
curl http://127.0.0.1:8102/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8/Qwen3.8-27B-UD-IQ2_M",
    "messages": [
      {"role": "user", "content": "你好，介绍一下你自己"}
    ],
    "stream": false,
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.8
  }'
```

### 流式文本请求

```bash
curl -N http://127.0.0.1:8102/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8/Qwen3.8-27B-UD-IQ2_M",
    "messages": [
      {"role": "user", "content": "写一段 Python 快速排序"}
    ],
    "stream": true
  }'
```

无工具时，流式为真正的逐 token SSE 流。

有工具时，服务会先进行非流式推理，再把完整结果包装成 SSE 事件返回，因此不是逐 token 流。

### 工具调用请求

```bash
curl http://127.0.0.1:8102/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8/Qwen3.8-27B-UD-IQ2_M",
    "messages": [
      {"role": "user", "content": "北京天气怎么样？"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询指定城市天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "城市名"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "stream": false
  }'
```

响应示例：

```json
{
  "id": "chatcmpl-xxxxxxxxxxxx",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "qwen3.8/Qwen3.8-27B-UD-IQ2_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xxxxxxxx",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120
  }
}
```

---

## Responses API

### 非流式文本请求

```bash
curl http://127.0.0.1:8102/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8/Qwen3.8-27B-UD-IQ2_M",
    "input": "你好",
    "stream": false
  }'
```

### 工具调用请求

```bash
curl http://127.0.0.1:8102/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8/Qwen3.8-27B-UD-IQ2_M",
    "input": [
      {
        "role": "user",
        "content": "北京天气怎么样？"
      }
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "查询指定城市天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": ["city"]
        }
      }
    ],
    "stream": false
  }'
```

Responses API 中，工具调用会以 `function_call` 输出项返回：

```json
{
  "id": "resp_xxxxxxxx",
  "object": "response",
  "status": "completed",
  "output": [
    {
      "id": "call_xxxxxxxx",
      "type": "function_call",
      "status": "completed",
      "name": "get_weather",
      "arguments": "{\"city\": \"北京\"}",
      "call_id": "call_xxxxxxxx"
    }
  ]
}
```

输入侧也支持：

- `function_call`
- `function_call_output`
- 普通 `message`
- `developer` 角色会转换为 `system`

---

## 工具预筛选机制

当请求中的工具数量超过 12 个时，服务会调用 `_select_relevant_tools` 进行筛选。

规则：

1. 核心工具永远保留：
   - `glob`
   - `grep`
   - `edit`
   - `pwsh`
   - `ask_user_question`
   - `present`

2. 根据用户消息中的关键词匹配工具组：
   - `goal`：目标、计划、继续等
   - `agent`：agent、子代理、委派等
   - `job`：后台任务、异步、任务队列等
   - `env`：依赖、环境、Python、Node、pnpm 等

3. 工具名直接出现在用户消息中，则强制保留。

4. 总数不超过 `max_tools`，默认 12。

可修改位置：

- `_CORE_TOOLS`
- `_TOOL_GROUPS`
- `_INTENT_KEYWORDS`
- `_select_relevant_tools(..., max_tools=12)`

---

## 工具调用解析细节

服务内部对模型输出做了大量兼容处理：

- `_normalize_tool_markers`
  - 统一全角竖线 `｜`
  - 统一制表竖线 `│`
  - 统一数学 divides `∣`
  - 统一 SentencePiece 下划线 `▁`

- `_extract_balanced_json`
  - 括号平衡提取 JSON
  - 修复嵌套对象被正则截断的问题

- `_ensure_valid_json`
  - 修复非法转义，例如 `\s`、`\m`
  - 解析失败时退化为 `{}`

- `_normalize_tool_call`
  - 兼容 `function.parameters` 与 `function.arguments`
  - 兼容 dict / list / str 参数
  - 自动补全 `id` 与 `type`

- `_sanitize_messages_for_template`
  - 确保 assistant 消息里的 `tool_calls[].function.arguments` 是 dict
  - 适配 Qwen3.8 GGUF `chat_template`

- `_strip_think`
  - 剥离 `<think>`、`<|thinking_section|>` 等标签

- `_postprocess_completion`
  - 非流式后处理入口
  - 仅在 `tools_enabled=True` 时解析工具调用

---

## MongoDB 日志

MongoDB 为可选功能。

如果安装了 `pymongo` 且连接成功，服务会记录：

- 请求历史：`request_history_gguf`
- 对话历史：`chat_history_gguf`

每条记录会附带：

```json
{
  "create_date": "2026-01-01 12:00:00"
}
```

如果未安装 `pymongo` 或连接失败，只会警告一次，不影响模型服务。

---

## Gradio 界面

代码中包含 `create_gradio_app()`，但默认未挂载，仅作占位。

如需启用，取消 `main()` 中以下注释：

```python
# gradio_blocks = create_gradio_app()
# app = gr.mount_gradio_app(app, gradio_blocks, path="/gradio")
```

然后访问：

```text
http://127.0.0.1:8102/gradio
```

当前 Gradio 页面未绑定事件，实际调用仍建议走 OpenAI 兼容接口。

---

## 注意事项

1. **只有客户端传 `tools` 才会解析工具调用。**
   否则模型可能模仿输出工具标记，但服务会按纯文本返回。

2. **`CHAT_FORMATS_WITHOUT_TOOLS` 当前包含 `qwen`。**
   如果你的模型实际 `chat_format` 是 `qwen`，且请求携带 `tools`，服务会抛出“不支持工具调用”的错误。请根据实际模型模板调整该集合，或设置支持工具调用的 `chat_format`。

3. **推理线程池为单 worker。**
   当前实现适合本地单模型低并发场景，不适合高并发生产环境。

4. **流式 + 工具调用不是逐 token 流。**
   有 `tools` 时，服务先非流式推理，再模拟 SSE 输出。

5. **CORS 默认允许所有来源。**
   生产环境请收紧 `allow_origins`。

6. **`MAX_CONTEXT` 较大。**
   默认 `32048 * 2 = 64096`，可能占用大量显存或内存。如遇 OOM，可降低：
   - `MAX_CONTEXT`
   - `n_batch`
   - `n_gpu_layers`
   - 或启用 `type_k` / `type_v` KV Cache 量化

7. **模型路径包含斜杠。**
   默认 `MODEL_NAME` 为 `qwen3.8/Qwen3.8-27B-UD-IQ2_M`，因此模型文件位于 `MODEL_DIR/qwen3.8/` 子目录下。

8. **空响应会重试一次。**
   非流式请求中，如果模型返回空内容且无工具调用，服务会提高 temperature 重试一次；仍为空则返回 502。

---

## 常见问题

### 1. 模型没有返回工具调用

检查：

- 客户端是否传入 `tools`
- `tool_choice` 是否为 `auto` 或 `required`
- `chat_format` 是否支持工具调用
- 模型是否确实输出 XML / 旧 Qwen / 裸 JSON 工具调用格式
- 查看服务日志中的 `[REQ DEBUG]` 与 `Parsed ... tool_calls`

### 2. 报错：当前 chat_format 不支持工具调用

原因：`CHAT_FORMATS_WITHOUT_TOOLS` 中包含当前 `chat_format`。

处理：

- 修改 `CHAT_FORMATS_WITHOUT_TOOLS`
- 或设置 `CHAT_FORMAT` 为支持工具调用的格式
- 或确认 GGUF 元数据中的 `chat_template` 是否支持工具调用

### 3. 显存或内存不足

尝试：

- 降低 `MAX_CONTEXT`
- 降低 `n_batch`
- 调整 `n_gpu_layers`
- 使用更小量化模型
- 启用 KV Cache 量化：
  ```python
  type_k=GGML_TYPE_Q8_0,
  type_v=GGML_TYPE_Q4_0,
  ```

### 4. MongoDB 连接失败

如果不需要日志，可忽略。警告只会出现一次。

如需启用：

```bash
pip install pymongo
```

并确认 MongoDB 服务已启动。

### 5. 流式工具调用不是逐 token

这是预期行为。有工具时，服务需要先拿到完整结构化工具调用，再包装为 SSE。

---

## 开发与扩展

### 自定义核心工具

修改：

```python
_CORE_TOOLS = {
    "glob", "grep", "edit", "pwsh",
    "ask_user_question", "present",
}
```

### 自定义工具组与意图关键词

修改：

```python
_TOOL_GROUPS = {
    "goal": {...},
    "agent": {...},
    "job": {...},
    "env": {...},
}
```

以及：

```python
_INTENT_KEYWORDS = {
    "goal": [...],
    "agent": [...],
    "job": [...],
    "env": [...],
}
```

### 调整工具数量上限

修改 `_select_relevant_tools` 的 `max_tools` 参数，默认 12。

### 新增工具调用格式

在 `_postprocess_completion` 中增加解析分支，并复用：

- `_normalize_tool_call`
- `_ensure_valid_json`
- `_extract_balanced_json`

### 调整模型加载参数

修改 `ModelManager._load_model`：

```python
self._model = Llama(
    model_path=config.MODEL_PATH,
    n_ctx=config.MAX_CONTEXT,
    n_batch=64,
    n_gpu_layers=-1,
    offload_kqv=True,
    flash_attn=True,
    chat_format=config.CHAT_FORMAT,
    n_threads=6,
    use_mmap=True,
    use_mlock=False,
)
```

---

## 许可证

当前代码未声明许可证。请在 GitHub 仓库中根据你的发布意愿补充 `LICENSE` 文件。

---

## 致谢

本项目依赖并受益于以下开源项目：

- FastAPI
- Uvicorn
- llama-cpp-python
- Gradio
- Pydantic
- PyMongo
- PyTorch
- OpenAI API 兼容生态

---

## 一句话总结

这是一个面向本地 GGUF 模型的 OpenAI 兼容服务，重点解决工具调用解析、Qwen3.8 XML 工具模板兼容、OpenAI Chat Completions / Responses API 适配与本地低并发部署问题。
