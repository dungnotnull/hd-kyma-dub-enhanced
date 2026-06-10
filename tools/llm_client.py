"""Unified LLM client: Claude (primary) → OpenAI (fallback) → Ollama (offline)."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

COST_PER_1K: dict[str, dict[str, float]] = {
    "claude-opus-4-8":    {"input": 0.015,  "output": 0.075},
    "claude-sonnet-4-6":  {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "gpt-4o":             {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":        {"input": 0.00015,"output": 0.0006},
    "llama3":             {"input": 0.0,    "output": 0.0},
}

PROVIDER_GUIDANCE = {
    "claude":  "Long-context reasoning, script adaptation, idiomatic translation",
    "openai":  "Multimodal review, JSON structured output",
    "ollama":  "Privacy-sensitive data, offline mode, high-volume low-cost tasks",
}


class UnifiedLLMClient:
    """Claude/OpenAI/Ollama unified client with streaming and retry."""

    def __init__(
        self,
        provider_priority: Optional[list[str]] = None,
        memory_manager=None,
    ):
        self.provider_priority = provider_priority or self._build_provider_chain()
        self._memory = memory_manager

    def _build_provider_chain(self) -> list[str]:
        privacy_mode = os.getenv("PRIVACY_MODE", "").lower() in ("1", "true", "yes")
        if privacy_mode:
            return ["ollama"]
        chain = []
        if os.getenv("ANTHROPIC_API_KEY"):
            chain.append("claude")
        if os.getenv("OPENAI_API_KEY"):
            chain.append("openai")
        chain.append("ollama")
        return chain or ["ollama"]

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        task: str = "general",
    ) -> str:
        last_error = None
        for provider in self.provider_priority:
            try:
                result = await self._call_with_retry(
                    provider, prompt, system, max_tokens, temperature
                )
                self._log_cost(provider, task, prompt, result)
                return result
            except Exception as e:
                logger.warning("Provider %s failed (%s); trying next", provider, e)
                last_error = e
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        for provider in self.provider_priority:
            try:
                async for chunk in self._stream_provider(
                    provider, prompt, system, max_tokens, temperature
                ):
                    yield chunk
                return
            except Exception as e:
                logger.warning("Stream from %s failed (%s); trying next", provider, e)
        raise RuntimeError("All LLM providers failed for streaming")

    async def _call_with_retry(
        self, provider: str, prompt: str, system: str,
        max_tokens: int, temperature: float,
        max_attempts: int = 3,
    ) -> str:
        import asyncio as aio
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                if provider == "claude":
                    return await self._call_claude(prompt, system, max_tokens, temperature)
                elif provider == "openai":
                    return await self._call_openai(prompt, system, max_tokens, temperature)
                elif provider == "ollama":
                    return await self._call_ollama(prompt, system, max_tokens, temperature)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except Exception as e:
                if attempt == max_attempts:
                    raise
                logger.debug("Attempt %d/%d failed (%s); retrying in %.1fs", attempt, max_attempts, e, delay)
                await aio.sleep(delay)
                delay *= 2

    async def _call_claude(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        return resp.content[0].text

    async def _call_openai(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content

    async def _call_ollama(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> str:
        import aiohttp
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        payload = {
            "model": model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["response"]

    async def _stream_provider(
        self, provider: str, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> AsyncGenerator[str, None]:
        if provider == "claude":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            model = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
            kwargs = dict(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        else:
            result = await self._call_with_retry(
                provider, prompt, system, max_tokens, temperature
            )
            yield result

    def _log_cost(self, provider: str, task: str, prompt: str, result: str):
        if not self._memory:
            return
        model = _model_for_provider(provider)
        rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
        in_tokens = len(prompt.split()) * 1.3
        out_tokens = len(result.split()) * 1.3
        cost_usd = (in_tokens / 1000 * rates["input"]) + (out_tokens / 1000 * rates["output"])
        try:
            self._memory.log_llm_cost(
                provider=provider,
                model=model,
                task=task,
                input_tokens=int(in_tokens),
                output_tokens=int(out_tokens),
                cost_usd=cost_usd,
            )
        except Exception:
            pass

    def complete_sync(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Synchronous wrapper for use in non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self.complete(prompt, system, max_tokens)
                    )
                    return future.result(timeout=120)
            return loop.run_until_complete(self.complete(prompt, system, max_tokens))
        except Exception:
            return asyncio.run(self.complete(prompt, system, max_tokens))


def _model_for_provider(provider: str) -> str:
    if provider == "claude":
        return os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    return os.getenv("OLLAMA_MODEL", "llama3")
