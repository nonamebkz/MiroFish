"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.llm_client")

_THINKING_BLOCK_RE = re.compile(
    r"<think>([\s\S]*?)</think>", re.IGNORECASE
)


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _complete_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return (content or "").strip()

    @staticmethod
    def _strip_markdown_json_fence(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        return re.sub(
            r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE
        ).strip()

    def _try_parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a single JSON object from model output (markdown fence / trailing junk tolerant)."""
        if not text:
            return None
        candidate = self._strip_markdown_json_fence(text)
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        if start >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(candidate, start)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
        return None

    def _parse_llm_json_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Some providers wrap JSON in markdown fences, put JSON only inside
        <think>...</think>, or prepend thinking so that stripping
        leaves nothing — try several extraction strategies.
        """
        if not raw:
            return None

        parsed = self._try_parse_json_object(raw)
        if parsed is not None:
            return parsed

        for m in _THINKING_BLOCK_RE.finditer(raw):
            parsed = self._try_parse_json_object(m.group(1))
            if parsed is not None:
                return parsed

        remainder = self._strip_thinking_tags(raw)
        parsed = self._try_parse_json_object(remainder)
        if parsed is not None:
            return parsed

        return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本
        """
        content = self._complete_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(
            r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE
        ).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        raw = self._complete_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        result = self._parse_llm_json_response(raw)
        if result is not None:
            return result

        logger.warning(
            "LLM JSON parse failed with response_format=json_object; retrying without it."
        )
        raw = self._complete_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )
        result = self._parse_llm_json_response(raw)
        if result is not None:
            return result

        preview = (raw[:800] + "…") if len(raw) > 800 else raw
        raise ValueError(
            "LLM返回的JSON格式无效或内容为空。"
            f" raw长度={len(raw)}，预览={preview!r}"
        )
