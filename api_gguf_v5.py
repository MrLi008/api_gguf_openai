# 以下是修改后的完整代码：

# ```python
"""
OpenAI 兼容的 LLM 服务，支持工具调用 (Function Calling)。

重构要点：
  - 括号平衡的 JSON 提取，修复嵌套参数丢失问题
  - asyncio.get_running_loop()，兼容 Python 3.12+
  - 工具调用能力检查基于模型实际 chat_format
  - 修复流式 StopIteration → RuntimeError 的坑
  - 统一内容规范化 / think 剥离 / 工具调用解析
  - 响应 API 的 function_call 项去除多余的 function 字段
  - 降低 frequency/presence penalty，避免干扰工具调用
  - 【新增】支持 Qwen3.8 GGUF chat_template 的 XML 风格工具调用
    <tool_call><function=...><parameter=...>...</parameter></function></tool_call>

依赖：gradio, llama-cpp-python, fastapi, uvicorn, pydantic, pymongo（可选）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0
from pydantic import BaseModel, Field

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    MongoClient = None  # type: ignore


# ============================================================
# 配置
# ============================================================
class Config:
    MODEL_NAME = "qwen3.8/Qwen3.8-27B-UD-IQ2_M"
    # MODEL_NAME = "DeepSeek-R1-0528-Qwen3-8B-Q8_0"
    MODEL_DIR = os.getenv("MODEL_DIR", "E:/n_c_r_release/.cache/.gguf")
    MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.gguf")

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8102"))

    MAX_TOKENS_DEFAULT = 4096
    # MAX_CONTEXT = 24576
    MAX_CONTEXT = 32048*2
    TOP_P = 0.8
    TEMPERATURE = 0.6

    # MongoDB（可选）
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("DB_NAME", "my_code_db")
    COLLECTION_CHAT = "chat_history_gguf"
    COLLECTION_REQUEST = "request_history_gguf"

    # None → 从 GGUF 元数据自动检测 chat_template
    CHAT_FORMAT = os.getenv("CHAT_FORMAT", None)


config = Config()


# 已知不支持工具调用的 chat_format
CHAT_FORMATS_WITHOUT_TOOLS = {
    "qwen", "llama-2", "vicuna", "command-r", "gemma",
    "falcon", "mpt", "openchat", "baichuan", "chatglm", "rwkv",
}


# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 线程池 —— 同步推理搬到子线程，避免阻塞事件循环
# ============================================================
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-infer")

# 流式迭代哨兵（不能在协程中直接抛 StopIteration）
_STREAM_END = object()


async def run_inference(func, *args, **kwargs):
    """把同步推理放到线程池执行。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _inference_executor,
        lambda: func(*args, **kwargs),
    )


def _safe_next(gen):
    """线程池中安全地取下一项；结束时返回哨兵。"""
    try:
        return next(gen)
    except StopIteration:
        return _STREAM_END


# ============================================================
# MongoDB（单例）
# ============================================================
_mongo_client = None
_mongo_warned = False


def get_mongo_client():
    global _mongo_client
    if not MONGO_AVAILABLE:
        return None
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                config.MONGO_URI, serverSelectionTimeoutMS=2000
            )
        except Exception as e:
            logger.warning(f"MongoDB connect failed: {e}")
            return None
    return _mongo_client


def save_to_mongo(collection_name: str, data: dict) -> None:
    """异步安全的旁路日志；Mongo 不可用时只警告一次。"""
    global _mongo_warned
    client = get_mongo_client()
    if client is None:
        if not _mongo_warned:
            logger.warning("MongoDB not available; request logging disabled.")
            _mongo_warned = True
        return
    try:
        db = client[config.DB_NAME]
        doc = dict(data)
        doc["create_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        db[collection_name].insert_one(doc)
    except Exception as e:
        logger.warning(f"MongoDB error: {e}")


# ============================================================
# 文本 / JSON 工具
# ============================================================
# 匹配各类 think 标签变体（含全角 ｜ 和 ▁）
_THINK_BLOCK = re.compile(
    r"<[｜|]?\s*think(?:ing)?[▁_]?section\s*[｜|]?>"
    r".*?"
    r"<[｜|]?/\s*think(?:ing)?[▁_]?section\s*[｜|]?>"
    r"|<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)

# 孤立标签（被截断时）
_THINK_TAG = re.compile(
    r"<[｜|]?\s*/?\s*think(?:ing)?[▁_]?section\s*[｜|]?>",
    re.IGNORECASE,
)

# 旧 Qwen 标记 <|tool_calls_begin|>...</|tool_calls_end|>
_TOOL_CALLS_BLOCK = re.compile(
    r"<\|tool_calls_begin\|>(.*?)<\|tool_calls_end\|>", re.DOTALL
)

_TOOL_CALL_BEGIN = "<|tool_call_begin|>"
_TOOL_CALL_END = "<|tool_call_end|>"
_TOOL_SEP = "<|tool_sep|>"

# ============================================================
# Qwen3.8 GGUF chat_template 的 XML 工具调用解析
# 格式：
# <tool_call>
# <function=name>
# <parameter=key>
# value
# </parameter>
# </function>
# </tool_call>
# ============================================================
_XML_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

_XML_FUNCTION_BLOCK = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)

_XML_PARAMETER_BLOCK = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)

# 合法 JSON 反斜杠转义
_VALID_ESCAPE_CHARS = '"\\/bfnrtu'
# ============================================================
# 工具预筛选
# ============================================================
# 永远保留的核心工具（文件操作 + 用户交互）
_CORE_TOOLS = {
    "glob", "grep", "edit", "pwsh",
    "ask_user_question", "present",
}

# 工具类别分组 —— 用于意图识别
_TOOL_GROUPS = {
    "goal":       {"create_goal", "get_goal", "update_goal", "exit_plan_mode"},
    "agent":      {"interrupt_agent", "list_agents", "send_message", "create_agent"},
    "job":        {"job_kill", "job_list", "job_output"},
    "env":        {"load_workspace_dependencies"},
}

# 意图关键词 → 工具组
_INTENT_KEYWORDS = {
    "goal":  ["goal", "objective", "目标", "计划", "plan", "round", "继续"],
    "agent": ["agent", "subagent", "子agent", "子代理", "delegate", "委派"],
    "job":   ["job", "background", "后台", "异步", "任务队列"],
    "env":   ["dependency", "dependencies", "依赖", "环境", "python", "node", "pnpm"],
}

def _normalize_tool_markers(s: str) -> str:
    """统一模型输出的各种"伪标记"变体。

    Qwen 词表里的特殊 token 解码后可能带出：
      - ｜ U+FF5C 全角竖线
      - │ U+2502 制表竖线
      - ∣ U+2223 数学 divides
      - ▁ U+2581 SentencePiece 词边界（视觉上像下划线）

    ▁ 在正文里几乎只出现在特殊 token 内部，全量替换是安全的；
    不替换会导致 <|tool_calls_begin|> 匹配失败，工具调用退化为纯文本。
    """
    if not s:
        return s
    return (
        s.replace("｜", "|")
         .replace("│", "|")
         .replace("∣", "|")
         .replace("▁", "_")     # ← 关键：必须加回来
    )


def _fix_invalid_escapes(s: str) -> str:
    """修复 JSON 字符串中的非法转义（\\s、\\m 等）。"""
    if not s:
        return s
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", s)


def _strip_think(content: Optional[str]) -> Optional[str]:
    """剥离各类 think 标签（含 Qwen3 的 <|thinking_section|> 及其变体）。"""
    if not content:
        return content
    content = _normalize_tool_markers(content)  # 先把 ｜▁ 归一
    content = _THINK_BLOCK.sub("", content)
    content = _THINK_TAG.sub("", content)
    content = content.replace("<think>", "").replace("</think>", "")
    return content.strip()


def _ensure_valid_json(s: str) -> str:
    """尝试解析 JSON；失败时修复非法转义；最终失败返回 '{}'。"""
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    fixed = _fix_invalid_escapes(s)
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        logger.warning(f"Invalid tool arguments, force to {{}}: {s[:200]!r}")
        return "{}"


def _extract_balanced_json(s: str, start: int = 0) -> Optional[Tuple[str, int]]:
    """从 start 处开始提取第一个括号平衡的 {...}。

    正确处理字符串字面量与转义 —— 修复原正则 `\\{.*?\\}` 遇到嵌套
    对象就截断的 bug。

    返回 (json_str, end_index_exclusive) 或 None。
    """
    idx = s.find("{", start)
    if idx == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(idx, len(s)):
        c = s[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[idx:i + 1], i + 1
    return None


def _normalize_content(content: Any) -> str:
    """OpenAI 多模态 content → 纯文本字符串。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype in ("text", "input_text", "output_text"):
                    texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return str(content)


def log_pretty(title: str, data: Any, max_len: int = 2000) -> None:
    try:
        text = json.dumps(data, ensure_ascii=False, default=str, indent=2)
    except Exception:
        text = str(data)
    if len(text) > max_len:
        text = text[:max_len] + f"\n... [truncated, total {len(text)} chars]"
    logger.info(f"===== {title} =====\n{text}\n===== END {title} =====")


# ============================================================
# 工具调用解析
# ============================================================
def _normalize_tool_call(tc: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """单个 tool_call → OpenAI 标准结构。

    兼容：
      - function.parameters → function.arguments
      - arguments 为 dict → JSON 字符串
      - 缺 id / type → 自动补全
      - 参数 JSON 非法 → 修复或退化为 {}
    """
    fn = tc.get("function") or {}

    args_value = fn.get("arguments")
    if args_value is None and "parameters" in fn:
        args_value = fn["parameters"]

    if args_value is None:
        args_str = "{}"
    elif isinstance(args_value, str):
        args_str = args_value
    elif isinstance(args_value, (dict, list)):
        try:
            args_str = json.dumps(args_value, ensure_ascii=False, sort_keys=True)
        except Exception:
            args_str = "{}"
    else:
        args_str = str(args_value)

    args_str = _ensure_valid_json(args_str)

    tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
    return {
        "id": tc_id,
        "type": tc.get("type", "function"),
        "function": {
            "name": fn.get("name", ""),
            "arguments": args_str,
        },
    }

def _args_str_to_obj(args: Any) -> Any:
    """把 JSON 字符串参数解析成对象；已是对象则原样返回。"""
    if args is None:
        return {}
    if isinstance(args, (dict, list)):
        return args
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return json.loads(_fix_invalid_escapes(s))
            except json.JSONDecodeError:
                logger.warning(
                    f"tool_call arguments not valid JSON, fallback to {{}}: "
                    f"{s[:200]!r}"
                )
                return {}
    return {}


def _sanitize_messages_for_template(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """确保 assistant 消息里的 tool_calls[].function.arguments 是 dict。

    Qwen3.8 GGUF chat_template 对字符串 arguments 会 raise_exception，
    这里是进入 create_chat_completion 前的统一清洗。
    """
    if not messages:
        return messages

    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue

        tcs = msg.get("tool_calls")
        if not tcs or not isinstance(tcs, list):
            sanitized.append(msg)
            continue

        new_msg = dict(msg)
        new_tcs: List[Dict[str, Any]] = []
        for tc in tcs:
            if not isinstance(tc, dict):
                new_tcs.append(tc)
                continue
            tc_copy = dict(tc)
            fn = dict(tc_copy.get("function") or {})
            fn["arguments"] = _args_str_to_obj(fn.get("arguments"))
            tc_copy["function"] = fn
            tc_copy.setdefault("type", "function")
            tc_copy.setdefault("id", f"call_{uuid.uuid4().hex[:8]}")
            new_tcs.append(tc_copy)
        new_msg["tool_calls"] = new_tcs
        sanitized.append(new_msg)

    return sanitized
def _parse_one_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """解析单个工具调用片段（已被 <|tool_call_begin|> 切分）。"""
    t = text.strip()
    if not t:
        return None

    result = _extract_balanced_json(t)
    if not result:
        return None
    json_str, _ = result

    json_start = t.find(json_str)
    if json_start < 0:
        return None
    prefix = t[:json_start]

    # 清理前缀里的各种标记
    prefix = prefix.replace(_TOOL_CALL_END, " ").replace(_TOOL_SEP, " ")
    prefix = re.sub(r"^function\s+", "", prefix.strip(), flags=re.IGNORECASE)
    prefix = prefix.strip().strip("<>").strip()

    tokens = prefix.split()
    if not tokens:
        return None
    name = tokens[0].strip("<>")
    if not re.match(r"^[\w\.\-]+$", name):
        return None

    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": _ensure_valid_json(json_str)},
    }


def parse_tool_calls_from_content(
    content: str,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """解析带 <|tool_calls_begin|> 标记的 content。

    返回 (clean_content, tool_calls_or_None)。
    """
    normalized = _normalize_tool_markers(content)
    if not normalized or "<|tool_calls_begin|>" not in normalized:
        return normalized, None

    tool_calls: List[Dict[str, Any]] = []

    def _consume_block(match: re.Match) -> str:
        inner = match.group(1)
        # 优先按 begin 切分；退化到按 end 切分
        if _TOOL_CALL_BEGIN in inner:
            chunks = inner.split(_TOOL_CALL_BEGIN)
        else:
            chunks = [inner]

        for chunk in chunks:
            if not chunk.strip():
                continue
            # 一个 chunk 中可能包含多个 end 分隔的调用
            for sub in chunk.split(_TOOL_CALL_END):
                if not sub.strip():
                    continue
                tc = _parse_one_tool_call(sub)
                if tc:
                    tool_calls.append(tc)
        return ""

    clean = _TOOL_CALLS_BLOCK.sub(_consume_block, normalized).strip()
    return clean, (tool_calls if tool_calls else None)


def _extract_xml_functions(inner: str, out: List[Dict[str, Any]]) -> None:
    """从 <tool_call> 内部或裸 XML 中提取 <function=...> 块。"""
    for fm in _XML_FUNCTION_BLOCK.finditer(inner):
        name = fm.group(1).strip()
        body = fm.group(2)
        args: Dict[str, Any] = {}

        for pm in _XML_PARAMETER_BLOCK.finditer(body):
            key = pm.group(1).strip()
            raw_val = pm.group(2).strip()

            # 尝试把参数值解析成 JSON 类型；失败则保留字符串
            try:
                val = json.loads(raw_val)
            except Exception:
                val = raw_val
            args[key] = val

        out.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })


def parse_xml_tool_calls_from_content(
    content: str,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """解析 Qwen3.8 GGUF chat_template 的 XML 工具调用。

    返回 (clean_content, tool_calls_or_None)。
    """
    normalized = _normalize_tool_markers(content)
    if not normalized:
        return normalized, None

    lower = normalized.lower()
    if "<tool_call>" not in lower and "<function=" not in lower:
        return normalized, None

    tool_calls: List[Dict[str, Any]] = []

    # 优先按 <tool_call>...</tool_call> 块解析
    def _consume_block(match: re.Match) -> str:
        inner = match.group(1)
        _extract_xml_functions(inner, tool_calls)
        return ""

    clean = _XML_TOOL_CALL_BLOCK.sub(_consume_block, normalized)

    # 如果没有 <tool_call> 包裹，但存在 <function=...>，则直接解析裸函数块
    if not tool_calls and "<function=" in lower:
        _extract_xml_functions(normalized, tool_calls)
        clean = _XML_FUNCTION_BLOCK.sub("", normalized)

    clean = clean.strip()
    return clean, (tool_calls if tool_calls else None)


def _try_parse_bare_json_toolcall(content: str) -> Optional[List[Dict[str, Any]]]:
    """识别"裸 JSON 工具调用"（无任何标记）。

    形态 1：{"type": "function", "function": {...}}
    形态 2：{"name": "...", "arguments"/"parameters": ...}
    也可能被 ```json ... ``` 包裹。
    """
    if not content or not content.strip():
        return None

    text = content.strip()

    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = json.loads(_fix_invalid_escapes(text))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None

    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        if isinstance(fn.get("name"), str) and fn["name"]:
            return [{"type": "function", "function": fn}]

    if isinstance(obj.get("name"), str) and obj["name"]:
        if "arguments" in obj or "parameters" in obj:
            return [{"type": "function", "function": obj}]

    return None


def _postprocess_completion(
    completion: Any, tools_enabled: bool = False
) -> Any:
    """非流式 completion 后处理：think 剥离 + tool_calls 标准化。

    tools_enabled=False 时不做任何工具调用识别，避免把代码示例误判。
    """
    if completion is None:
        return completion
    if hasattr(completion, "model_dump"):
        completion = completion.model_dump()
    elif hasattr(completion, "dict"):
        completion = completion.dict()
    if not isinstance(completion, dict):
        return completion

    try:
        for choice in completion.get("choices") or []:
            msg = choice.get("message") or {}

            # 1) 剥离 think
            if isinstance(msg.get("content"), str):
                msg["content"] = _strip_think(msg["content"])

            if not tools_enabled:
                continue

            # 2) 已是结构化 tool_calls
            if msg.get("tool_calls"):
                msg["tool_calls"] = [
                    _normalize_tool_call(tc, i)
                    for i, tc in enumerate(msg["tool_calls"])
                ]
                if not (msg.get("content") or "").strip():
                    msg["content"] = None
                choice["finish_reason"] = "tool_calls"
                continue

            content = msg.get("content") or ""
            if not content.strip():
                continue

            normalized = _normalize_tool_markers(content)

            # 3) Qwen3.8 GGUF chat_template 的 XML 工具调用
            #    <tool_call><function=...><parameter=...>...</parameter></function></tool_call>
            if "<tool_call>" in normalized.lower() or "<function=" in normalized.lower():
                clean, parsed = parse_xml_tool_calls_from_content(content)
                if parsed:
                    msg["tool_calls"] = [
                        _normalize_tool_call(tc, i)
                        for i, tc in enumerate(parsed)
                    ]
                    msg["content"] = (clean or "").strip() or None
                    choice["finish_reason"] = "tool_calls"
                    logger.info(
                        f"Parsed {len(parsed)} XML tool_calls: "
                        f"{[t['function']['name'] for t in msg['tool_calls']]}"
                    )
                else:
                    logger.warning(
                        f"Found XML tool markers but no calls parsed; "
                        f"head={content[:200]!r}"
                    )
                continue

            # 4) 旧 Qwen 标记 <|tool_calls_begin|>
            if "<|tool_calls_begin|>" in normalized:
                clean, parsed = parse_tool_calls_from_content(content)
                if parsed:
                    msg["tool_calls"] = [
                        _normalize_tool_call(tc, i)
                        for i, tc in enumerate(parsed)
                    ]
                    msg["content"] = (clean or "").strip() or None
                    choice["finish_reason"] = "tool_calls"
                    logger.info(
                        f"Parsed {len(parsed)} tool_calls: "
                        f"{[t['function']['name'] for t in msg['tool_calls']]}"
                    )
                else:
                    stripped = (clean or "").strip()
                    if not stripped:
                        logger.warning(
                            f"Tool block stripped but no calls parsed; "
                            f"keeping original. head={content[:200]!r}"
                        )
                        msg["content"] = content
                    else:
                        msg["content"] = stripped
                continue

            # 5) 裸 JSON
            bare = _try_parse_bare_json_toolcall(content)
            if bare:
                msg["tool_calls"] = [
                    _normalize_tool_call(tc, i)
                    for i, tc in enumerate(bare)
                ]
                msg["content"] = None
                choice["finish_reason"] = "tool_calls"
                logger.info(
                    f"Parsed bare JSON tool_call: "
                    f"{msg['tool_calls'][0]['function']['name']}"
                )
    except Exception as e:
        logger.warning(f"Postprocess tool_calls failed: {e}")

    return completion


def _normalize_tools_for_chat_completion(
    tools_raw: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Responses API 风格 tools → Chat Completions 风格。"""
    if not tools_raw:
        return None
    normalized: List[Dict[str, Any]] = []
    for tool in tools_raw:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            normalized.append(tool)
            continue
        tool_type = tool.get("type", "function")
        fn: Dict[str, Any] = {}
        if "name" in tool:
            fn["name"] = tool["name"]
        if "description" in tool:
            fn["description"] = tool["description"]
        if "parameters" in tool:
            fn["parameters"] = tool["parameters"]
        if fn:
            normalized.append({"type": tool_type, "function": fn})
    return normalized or None

def _select_relevant_tools(
    tools: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    max_tools: int = 12,
) -> List[Dict[str, Any]]:
    """按用户最后一条消息的意图筛选工具。

    规则：
      1. 核心工具永远保留
      2. 通过意图关键词匹配工具组
      3. 工具名直接出现在用户消息中 → 强制保留
      4. 总数不超过 max_tools；超出时按相关性排序
    """
    if not tools or len(tools) <= max_tools:
        return tools

    # 取所有 user 消息（最后一条权重最高）
    user_texts: List[str] = []
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                user_texts.append(c)
    query = " ".join(user_texts).lower() if user_texts else ""

    # 命中的工具组
    active_groups: set = set()
    if query:
        for group, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query:
                    active_groups.add(group)
                    break

    # 强制保留的工具名集合
    must_keep: set = set(_CORE_TOOLS)
    for group in active_groups:
        must_keep |= _TOOL_GROUPS.get(group, set())

    # 工具名直接出现在 query 中 → 强制保留
    for t in tools:
        name = (t.get("function") or {}).get("name", "")
        if name and name.lower() in query:
            must_keep.add(name)

    # 分拣
    selected: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for t in tools:
        name = (t.get("function") or {}).get("name", "")
        if name in must_keep:
            selected.append(t)
        else:
            others.append(t)

    # 如果还没到上限，按工具名/描述与 query 的相关性补齐
    if len(selected) < max_tools and others:
        def _score(t: Dict[str, Any]) -> int:
            fn = t.get("function") or {}
            name = (fn.get("name") or "").lower()
            desc = (fn.get("description") or "").lower()
            s = 0
            for tok in re.split(r"[_\W]+", name):
                if len(tok) >= 3 and tok in query:
                    s += 20
            for tok in re.findall(r"[a-zA-Z\u4e00-\u9fff]{3,}", desc[:200]):
                if tok.lower() in query:
                    s += 1
            return s
        others.sort(key=_score, reverse=True)
        selected.extend(others[: max_tools - len(selected)])

    logger.info(
        f"Tool filter: {len(tools)} → {len(selected)} "
        f"(groups={sorted(active_groups)}, core={len(_CORE_TOOLS)})"
    )
    return selected
# ============================================================
# 模型管理
# ============================================================
class ModelManager:
    def __init__(self) -> None:
        self._model: Optional[Llama] = None

    @property
    def model(self) -> Llama:
        if self._model is None:
            self._load_model()
        assert self._model is not None
        return self._model

    def _load_model(self) -> None:
        logger.info(f"Loading model from {config.MODEL_PATH}...")
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")
        self._model = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.MAX_CONTEXT,
            n_batch=64,
            n_gpu_layers=-1,
            offload_kqv=True,
            # offload_kqv=False,
            flash_attn=True,
            chat_format=config.CHAT_FORMAT,
            n_threads=6,
            use_mmap=True,
            use_mlock=False,

            # type_k=GGML_TYPE_Q8_0,   # K cache 从 f16 压到 q8_0
            # type_v=GGML_TYPE_Q4_0,   # V cache 同理
        )
        actual_fmt = getattr(self._model, "chat_format", None)
        logger.info(f"Model loaded. chat_format={actual_fmt}")
        meta_template = self._model.metadata.get(
            "tokenizer.chat_template", "NOT FOUND"
        )
        logger.info(f"GGUF chat_template (head): {str(meta_template)[:200]}")

    def warmup(self) -> None:
        """幂等预热。"""
        if self._model is None:
            self._load_model()

    # --------------------------------------------------------
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = config.MAX_TOKENS_DEFAULT,
        top_p: float = config.TOP_P,
        temperature: float = config.TEMPERATURE,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Union[Generator[str, None, None], Dict[str, Any]]:
        model = self.model

        kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "stream": stream,
            # 工具调用场景下高惩罚会破坏结构化输出
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        if tools:
            kwargs["tools"] = tools
            if not tool_choice:
                tool_choice = "auto"

            # 基于模型实际的 chat_format 检查
            actual_fmt = (getattr(model, "chat_format", None) or "").lower()
            if actual_fmt and actual_fmt in CHAT_FORMATS_WITHOUT_TOOLS:
                raise ValueError(
                    f"当前 chat_format='{actual_fmt}' 不支持工具调用，"
                    f"请设置 CHAT_FORMAT=qwen（或其它支持工具调用的格式）后重启。"
                )
            logger.info(
                f"Tool calling enabled: chat_format={actual_fmt}, tools={len(tools)}"
            )

        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        # ★ 最后一道防线：arguments 必须是 dict，Qwen3.8 chat_template 才接受
        kwargs["messages"] = _sanitize_messages_for_template(kwargs["messages"])
        
        with torch.no_grad():
            
            completion = model.create_chat_completion(**kwargs)

        if stream:
            return self._stream_response(completion)

        # 非流式：始终后处理（think 剥离），仅在 tools 存在时解析工具
        return _postprocess_completion(completion, tools_enabled=bool(tools))

    def _stream_response(
        self,
        completion_iter: Generator,
        model_name: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """llama-cpp-python 流 → OpenAI SSE。

        注意：仅用于【无 tools】的纯文本流式。
        """
        created = int(time.time())
        model_name = model_name or config.MODEL_NAME

        for chunk in completion_iter:
            if isinstance(chunk, dict):
                pass
            elif hasattr(chunk, "model_dump"):
                chunk = chunk.model_dump()
            elif hasattr(chunk, "dict"):
                chunk = chunk.dict()
            else:
                chunk = {"choices": [{}]}

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            payload = {
                "id": chunk.get("id", "chatcmpl-stream"),
                "object": "chat.completion.chunk",
                "created": chunk.get("created", created),
                "model": model_name,
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": delta,
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    def clear_cache(self) -> None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


model_manager = ModelManager()


# ============================================================
# SSE / Responses 辅助
# ============================================================
def non_stream_to_sse(
    response: Dict[str, Any], model_name: Optional[str] = None
) -> Generator[str, None, None]:
    """非流式 completion → OpenAI SSE（带 tools 的流式请求走这条路）。"""
    created = int(time.time())
    model_name = model_name or config.MODEL_NAME

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason", "stop")

    if finish_reason == "length":
        usage = response.get("usage") or {}
        logger.warning(
            "Response truncated due to max_tokens! "
            f"prompt_tokens={usage.get('prompt_tokens')}, "
            f"completion_tokens={usage.get('completion_tokens')}"
        )

    def make_chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
        payload = {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # 1) 角色
    yield make_chunk({"role": "assistant"})

    # 2) 正文
    content = message.get("content")
    if content:
        yield make_chunk({"content": content})

    # 3) tool_calls
    tool_calls = message.get("tool_calls")
    if tool_calls:
        yield make_chunk({
            "tool_calls": [
                {
                    "index": i,
                    "id": tc.get("id") or f"call_{i}",
                    "type": tc.get("type", "function"),
                    "function": tc.get("function", {}),
                }
                for i, tc in enumerate(tool_calls)
            ]
        })

    # 4) 结束
    yield make_chunk({}, finish_reason)
    yield "data: [DONE]\n\n"


def _responses_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _responses_input_to_messages(input_data: Any) -> List[Dict[str, Any]]:
    """Responses API input → Chat Completions messages。"""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if not isinstance(input_data, list):
        return []

    messages: List[Dict[str, Any]] = []
    for item in input_data:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        # function_call → assistant 消息
        if item_type == "function_call":
            call_id = item.get("id") or item.get("call_id") or ""
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }],
            })
            continue

        # function_call_output → tool 消息
        if item_type == "function_call_output":
            output = item.get("output", "")
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "",
                "content": str(output),
            })
            continue

        # 普通消息
        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        msg: Dict[str, Any] = {
            "role": role,
            "content": _normalize_content(item.get("content")),
        }
        if role == "tool" and item.get("tool_call_id"):
            msg["tool_call_id"] = item["tool_call_id"]
        messages.append(msg)

    logger.info(f"Input conversion: {len(messages)} messages generated")
    log_pretty("Converted Messages", messages)
    return messages


def _extract_response_text(completion: Dict[str, Any]) -> str:
    try:
        choice = completion["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)
    except Exception:
        return ""


def _build_response_object(
    resp_id: str,
    model: str,
    created_at: int,
    status: str,
    output: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    resp: Dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output or [],
    }
    if usage:
        resp["usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    else:
        resp["usage"] = None
    return resp


def _chat_tool_calls_to_responses_output(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Chat 风格 tool_calls → Responses API output 项（去掉多余 function 字段）。"""
    output_items: List[Dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        call_id = tc.get("id") or f"call_{i}"
        output_items.append({
            "id": call_id,
            "type": "function_call",
            "status": "completed",
            "name": name,
            "arguments": args,
            "call_id": call_id,
        })
    return output_items


def _responses_stream(
    resp_id: str,
    model: str,
    text: str,
    tool_calls: Optional[List[Dict[str, Any]]],
    created_at: int,
    usage: Optional[Dict[str, int]] = None,
) -> Generator[str, None, None]:
    """把完整响应打包成 Responses SSE 事件序列（带 tools 的流式）。"""
    msg_id = "msg_" + (resp_id[5:] if resp_id.startswith("resp_") else resp_id)

    base_resp = _build_response_object(resp_id, model, created_at, "in_progress")
    yield _responses_sse(
        "response.created", {"type": "response.created", "response": base_resp}
    )

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            call_id = tc.get("id") or f"call_{i}"

            yield _responses_sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": i,
                    "item": {
                        "id": call_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "name": name,
                        "arguments": "",
                        "call_id": call_id,
                    },
                },
            )
            yield _responses_sse(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": call_id,
                    "output_index": i,
                    "delta": args,
                },
            )
            yield _responses_sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": call_id,
                    "output_index": i,
                    "arguments": args,
                },
            )
            yield _responses_sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": i,
                    "item": {
                        "id": call_id,
                        "type": "function_call",
                        "status": "completed",
                        "name": name,
                        "arguments": args,
                        "call_id": call_id,
                    },
                },
            )
    else:
        yield _responses_sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        yield _responses_sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        yield _responses_sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
        )
        yield _responses_sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        )
        yield _responses_sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                },
            },
        )

    final_output = (
        _chat_tool_calls_to_responses_output(tool_calls)
        if tool_calls
        else [{
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }]
    )
    final_resp = _build_response_object(
        resp_id, model, created_at, "completed", output=final_output, usage=usage
    )
    yield _responses_sse(
        "response.completed", {"type": "response.completed", "response": final_resp}
    )


def _responses_true_stream(
    resp_id: str,
    model: str,
    chat_stream_gen: Generator[str, None, None],
    created_at: int,
) -> Generator[str, None, None]:
    """逐 token 的 Responses 流式（无 tools）。"""
    msg_id = "msg_" + (resp_id[5:] if resp_id.startswith("resp_") else resp_id)

    base_resp = _build_response_object(resp_id, model, created_at, "in_progress")
    yield _responses_sse(
        "response.created", {"type": "response.created", "response": base_resp}
    )
    yield _responses_sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield _responses_sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    full_text = ""
    for line in chat_stream_gen:
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        delta = (data.get("choices") or [{}])[0].get("delta") or {}
        content = delta.get("content") or ""
        if content:
            full_text += content
            yield _responses_sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content,
                },
            )

    yield _responses_sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        },
    )
    yield _responses_sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": full_text, "annotations": []}
                ],
            },
        },
    )
    final_resp = _build_response_object(
        resp_id, model, created_at, "completed",
        output=[{
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": full_text, "annotations": []}
            ],
        }],
        usage=None,
    )
    yield _responses_sse(
        "response.completed", {"type": "response.completed", "response": final_resp}
    )


# ============================================================
# Pydantic 模型
# ============================================================
class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ToolChoiceEnum(str, Enum):
    auto = "auto"
    none = "none"
    required = "required"


class ToolChoice(BaseModel):
    type: str = "function"
    function: Dict[str, str]


class ChatMessageToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    # 支持多模态 content（list）以便客户端兼容
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = config.MAX_TOKENS_DEFAULT
    top_p: Optional[float] = config.TOP_P
    temperature: Optional[float] = config.TEMPERATURE
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    response_format: Optional[Dict[str, Any]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = config.MODEL_NAME
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None


# ============================================================
# 请求处理辅助
# ============================================================
async def _do_chat_completion(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    top_p: float,
    temperature: float,
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    response_format: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """非流式推理的常用封装。"""
    return await run_inference(
        model_manager.chat_completion,
        messages=messages,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        stream=False,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
    )


def _parse_response_message(
    response: Dict[str, Any],
) -> Tuple[ChatMessage, str]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    resp_msg = ChatMessage(
        role=message.get("role", "assistant"),
        content=message.get("content"),
    )
    if message.get("tool_calls"):
        resp_msg.tool_calls = [
            ChatMessageToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=tc["function"],
            )
            for tc in message["tool_calls"]
        ]
    return resp_msg, choice.get("finish_reason", "stop")


def _is_empty_response(msg: ChatMessage) -> bool:
    content = msg.content if isinstance(msg.content, str) else ""
    return not (content or "").strip() and not msg.tool_calls


# ============================================================
# FastAPI 应用
# ============================================================
def create_openai_app() -> FastAPI:
    app = FastAPI(title="OpenAI Compatible API with Tools", version="2.3.0")

    # --------------------------------------------------------
    # /v1/chat/completions
    # --------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        logger.info(
            f"Received request: {len(request.messages)} messages, "
            f"stream={request.stream}, tools={bool(request.tools)}"
        )

        # ─── DEBUG：确认客户端到底传了什么 ───
        logger.info(
            f"[REQ DEBUG] tools_raw={request.tools!r} "
            f"tool_choice={request.tool_choice!r} "
            f"model={request.model!r}"
        )
        if request.tools:
            for i, t in enumerate(request.tools):
                logger.info(
                    f"[REQ DEBUG] tool[{i}] type={t.type} "
                    f"name={t.function.name} "
                    f"params_keys={list((t.function.parameters or {}).keys())}"
                )
        else:
            logger.warning(
                "[REQ DEBUG] ⚠ 客户端未传 tools 字段；"
                "模型会退化为「模仿」工具标记输出纯文本，"
                "服务器不会进入工具解析分支。"
            )
        # 1) 消息转换（多模态 → 纯文本）
        messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            item: Dict[str, Any] = {
                "role": msg.role,
                "content": _normalize_content(msg.content),
            }
            if msg.tool_calls:
                item["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            messages.append(item)

        # 2) 工具定义 / 策略
        tools = [t.model_dump() for t in request.tools] if request.tools else None
        
        # ★ 工具预筛选：先按意图裁剪，再交给模型
        if tools and len(tools) > 12:
            tools = _select_relevant_tools(tools, messages, max_tools=12)
            
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None
        if request.tool_choice:
            tool_choice = (
                request.tool_choice
                if isinstance(request.tool_choice, str)
                else request.tool_choice.model_dump()
            )

        # 3) 参数兜底
        max_tokens = request.max_tokens or config.MAX_TOKENS_DEFAULT
        top_p = request.top_p if request.top_p is not None else config.TOP_P
        temperature = (
            request.temperature
            if request.temperature is not None
            else config.TEMPERATURE
        )

        save_to_mongo(
            config.COLLECTION_REQUEST,
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
                "stream": request.stream,
            },
        )

        # 4) 流式
        if request.stream:
            async def generate():
                try:
                    if tools:
                        logger.info("Stream + tools: non-stream then emit SSE")
                        response = await _do_chat_completion(
                            messages, max_tokens, top_p, temperature,
                            tools, tool_choice, request.response_format,
                        )
                        for line in non_stream_to_sse(response):
                            yield line
                    else:
                        logger.info("Stream (text only): true streaming")
                        stream_gen = await run_inference(
                            model_manager.chat_completion,
                            messages=messages,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            temperature=temperature,
                            stream=True,
                            tools=None,
                            tool_choice=None,
                            response_format=request.response_format,
                        )
                        while True:
                            line = await run_inference(_safe_next, stream_gen)
                            if line is _STREAM_END:
                                break
                            yield line
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        # 5) 非流式
        try:
            response = await _do_chat_completion(
                messages, max_tokens, top_p, temperature,
                tools, tool_choice, request.response_format,
            )
            resp_msg, finish_reason = _parse_response_message(response)

            logger.info(
                f"[RAW RESPONSE] finish_reason={finish_reason}, "
                f"content_repr={(resp_msg.content or '')[:120]!r}, "
                f"tool_calls_count={len(resp_msg.tool_calls or [])}"
            )

            if finish_reason == "length":
                logger.warning(
                    f"Response truncated due to max_tokens={max_tokens}"
                )

            # 6) 空响应兜底：重试一次
            if _is_empty_response(resp_msg):
                logger.warning("Empty response; retrying with higher temperature...")
                retry_temp = min(temperature + 0.2, 1.0)
                response = await _do_chat_completion(
                    messages, max_tokens, top_p, retry_temp,
                    tools, tool_choice, request.response_format,
                )
                resp_msg, finish_reason = _parse_response_message(response)

                if _is_empty_response(resp_msg):
                    logger.error("Retry also empty.")
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "Model returned empty response twice",
                            "finish_reason": finish_reason,
                        },
                    )

            save_to_mongo(
                config.COLLECTION_CHAT,
                {"messages": messages, "response": response},
            )

            return ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        message=resp_msg, finish_reason=finish_reason
                    )
                ],
                model=config.MODEL_NAME,
                usage=response.get("usage"),
            )
        except ValueError as e:
            logger.error(f"Invalid request: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            logger.error(f"Completion error: {e}")
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            model_manager.clear_cache()

    # --------------------------------------------------------
    # /v1/responses
    # --------------------------------------------------------
    @app.post("/v1/responses")
    async def responses_api(raw_request: Request):
        try:
            payload = await raw_request.json()
        except Exception:
            return JSONResponse(
                status_code=400, content={"error": "invalid json body"}
            )

        log_pretty("Responses API Raw Payload", payload)

        model = payload.get("model") or config.MODEL_NAME
        stream = bool(payload.get("stream", False))

        max_output_tokens = (
            payload.get("max_output_tokens")
            or payload.get("max_tokens")
            or config.MAX_TOKENS_DEFAULT
        )
        top_p = payload.get("top_p")
        if top_p is None:
            top_p = config.TOP_P
        temperature = payload.get("temperature")
        if temperature is None:
            temperature = config.TEMPERATURE

        messages = _responses_input_to_messages(payload.get("input"))
        if not messages:
            return JSONResponse(
                status_code=400, content={"error": "input is required"}
            )

        tools = _normalize_tools_for_chat_completion(payload.get("tools"))
        if tools and len(tools) > 12:
            tools = _select_relevant_tools(tools, messages, max_tools=12)
        tool_choice = payload.get("tool_choice")
        response_format = payload.get("response_format")

        log_pretty("Normalized Tools", tools)
        log_pretty("Messages for Chat Completion", messages)

        resp_id = "resp_" + uuid.uuid4().hex
        created_at = int(time.time())

        # 非流式
        if not stream:
            try:
                completion = await _do_chat_completion(
                    messages, max_output_tokens, top_p, temperature,
                    tools, tool_choice, response_format,
                )
                text = _extract_response_text(completion)
                tool_calls = (
                    completion.get("choices", [{}])[0]
                    .get("message", {})
                    .get("tool_calls")
                )
                usage = completion.get("usage")

                if tool_calls:
                    output = _chat_tool_calls_to_responses_output(tool_calls)
                else:
                    output = [{
                        "id": "msg_" + resp_id[5:],
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }]

                response_obj = _build_response_object(
                    resp_id, model, created_at, "completed",
                    output=output, usage=usage,
                )
                logger.info(f"Extracted text: {text!r}")
                logger.info(f"Extracted tool_calls: {tool_calls!r}")
                return JSONResponse(content=response_obj)
            except Exception as e:
                logger.error(f"Responses API error: {e}")
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        # 流式
        async def generate():
            try:
                if tools:
                    logger.info(
                        "Responses stream + tools: non-stream then SSE events"
                    )
                    completion = await _do_chat_completion(
                        messages, max_output_tokens, top_p, temperature,
                        tools, tool_choice, response_format,
                    )
                    text = _extract_response_text(completion)
                    tool_calls = (
                        completion.get("choices", [{}])[0]
                        .get("message", {})
                        .get("tool_calls")
                    )
                    usage = completion.get("usage")
                    logger.info(f"Extracted text: {text!r}")
                    logger.info(f"Extracted tool_calls: {tool_calls!r}")
                    for evt in _responses_stream(
                        resp_id, model, text, tool_calls, created_at, usage
                    ):
                        yield evt
                else:
                    logger.info("Responses stream (text only): true streaming")
                    chat_stream_gen = await run_inference(
                        model_manager.chat_completion,
                        messages=messages,
                        max_tokens=max_output_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        stream=True,
                        tools=None,
                        tool_choice=None,
                        response_format=response_format,
                    )
                    for evt in _responses_true_stream(
                        resp_id, model, chat_stream_gen, created_at
                    ):
                        yield evt
            except Exception as e:
                logger.error(f"Responses stream error: {e}")
                traceback.print_exc()
                yield _responses_sse(
                    "error", {"type": "error", "error": str(e)}
                )

        return StreamingResponse(generate(), media_type="text/event-stream")

    # --------------------------------------------------------
    # 其它路由
    # --------------------------------------------------------
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": config.MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }],
        }

    @app.get("/models")
    async def list_models_short():
        return await list_models()

    @app.post("/v1/completions")
    async def text_completions():
        return JSONResponse(
            status_code=404,
            content={"error": "only /v1/chat/completions is supported"},
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": model_manager._model is not None,
            "chat_format": getattr(model_manager._model, "chat_format", None),
        }

    return app


# ============================================================
# Gradio 界面（占位，不做事件绑定）
# ============================================================
def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="Chat with Tools") as demo:
        gr.HTML(
            f"<h1 align='center'>{config.MODEL_NAME} - Chat with Tool Calling</h1>"
        )
        chatbot = gr.Chatbot(label="Conversation")
        with gr.Row():
            with gr.Column(scale=3):
                user_input = gr.Textbox(
                    show_label=False, placeholder="输入消息...", lines=5
                )
                with gr.Row():
                    submit_btn = gr.Button("发送")
            with gr.Column(scale=1):
                clear_btn = gr.Button("清空对话")
                max_length_slider = gr.Slider(
                    1, config.MAX_CONTEXT,
                    value=config.MAX_TOKENS_DEFAULT, step=1,
                    label="最大生成长度",
                )
                top_p_slider = gr.Slider(
                    0, 1, value=config.TOP_P, step=0.01, label="Top P"
                )
                temp_slider = gr.Slider(
                    0.01, 1, value=config.TEMPERATURE, step=0.01, label="Temperature"
                )
        # UI 仅作占位，Trae 调用走 /v1/chat/completions 或 /v1/responses
    return demo


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    app = create_openai_app()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"--> {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(
            f"<-- {request.method} {request.url.path} | {response.status_code}"
        )
        return response

    # 404 兜底
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def catch_all(path: str, request: Request):
        body = await request.body()
        try:
            body_str = body.decode("utf-8")
        except UnicodeDecodeError:
            body_str = str(body)
        logger.warning(
            f"Unmatched: {request.method} /{path} | "
            f"body={body_str[:500]}{'...' if len(body_str) > 500 else ''}"
        )
        return JSONResponse(
            status_code=404, content={"detail": f"Path '/{path}' not found"}
        )

    # 全局异常兜底
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}:\n"
            f"{traceback.format_exc()}"
        )
        return JSONResponse(status_code=500, content={"error": str(exc)})

    # 启动预热
    try:
        model_manager.warmup()
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        logger.error(
            "Service will still start; requests may fail. Check MODEL_PATH."
        )

    # 如需挂载 Gradio，取消下面两行注释
    # gradio_blocks = create_gradio_app()
    # app = gr.mount_gradio_app(app, gradio_blocks, path="/gradio")

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
# ```

# ---

# ## 本次改动一览（相对上一版）

# | 序号 | 位置 | 改动 |
# |---|---|---|
# | 1 | 正则区 | 新增 `_XML_TOOL_CALL_BLOCK`、`_XML_FUNCTION_BLOCK`、`_XML_PARAMETER_BLOCK` 三条正则 |
# | 2 | 工具解析区 | 新增 `_extract_xml_functions()` 与 `parse_xml_tool_calls_from_content()` |
# | 3 | `_postprocess_completion` | 在"旧 Qwen 标记"分支之前插入 XML 工具调用分支（编号 3），旧分支顺延为 4，裸 JSON 变为 5 |

# ## 解析行为说明

# 对模型输出的：

# ```xml
# <tool_call>
# <function=get_weather>
# <parameter=city>
# 北京
# </parameter>
# <parameter=unit>
# celsius
# </parameter>
# </function>
# </tool_call>
# ```

# 会得到：

# ```json
# {
#   "id": "call_xxxxxxxx",
#   "type": "function",
#   "function": {
#     "name": "get_weather",
#     "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
#   }
# }
# ```

# 要点：

# - 参数值先尝试 `json.loads`，能解析成 int/float/bool/dict/list 就转；否则保留原始字符串。所以数字、布尔、嵌套对象都能正确还原。
# - `<tool_call>` 前的自然语言说明会被保留在 `content` 中；只剥离工具调用块。
# - 若模型没被强制 `tools` 参数、但"模仿"输出 XML，也**只有**客户端传了 `tools` 时才会进入工具解析（`tools_enabled=True`），否则纯文本原样返回，避免把代码示例误判。
# - 兼容旧的 `<|tool_calls_begin|>` 标记与裸 JSON 工具调用，三者优先级：XML → 旧标记 → 裸 JSON。