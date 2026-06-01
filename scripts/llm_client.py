#!/usr/bin/env python3
"""Small chat client shared by GEM request executors.

The default mode is dry-run: no network call is made. Set --execute and provide
provider credentials to call a local or remote LLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def call_openai_compatible(messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
    base_url = os.environ.get("GEM_LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("GEM_LLM_API_KEY", "")
    model = os.environ.get("GEM_LLM_MODEL", "")
    if not base_url or not api_key or not model:
        raise RuntimeError("Set GEM_LLM_BASE_URL, GEM_LLM_API_KEY, and GEM_LLM_MODEL.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    system_texts = []
    contents = []
    for message in messages:
        role = message.get("role", "user")
        text = message_text(message)
        if not text:
            continue
        if role == "system":
            system_texts.append(text)
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )
    if system_texts:
        system_prompt = "System instructions:\n" + "\n\n".join(system_texts)
        if contents and contents[0]["role"] == "user":
            contents[0]["parts"].insert(0, {"text": system_prompt})
        else:
            contents.insert(0, {"role": "user", "parts": [{"text": system_prompt}]})
    if not contents:
        raise RuntimeError("Gemini request has no text content.")
    return contents


def extract_gemini_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini response has no candidates: {body}")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if "text" in part]
    if not texts:
        raise RuntimeError(f"Gemini response has no text parts: {body}")
    return "".join(texts)


def call_gemini(messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL") or os.environ.get("GEM_LLM_MODEL") or "gemini-3.5-flash"
    base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before using --provider gemini.")
    model_path = model if model.startswith("models/") else f"models/{model}"
    payload = {
        "contents": gemini_contents(messages),
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        f"{base_url}/{model_path}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {detail[:800]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    return extract_gemini_text(body), body.get("usageMetadata", {})


def call_chat(messages: list[dict[str, str]], max_tokens: int, temperature: float, provider: str) -> tuple[str, dict[str, Any]]:
    if provider == "gemini":
        return call_gemini(messages, max_tokens=max_tokens, temperature=temperature)
    return call_openai_compatible(messages, max_tokens=max_tokens, temperature=temperature)


def call_model(prompt: str, provider: str, temperature: float, max_tokens: int | None) -> str:
    raw, _usage = call_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens or 2048,
        temperature=temperature,
        provider=provider,
    )
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--text", type=Path, help="Path to a text file used as {{text}}.")
    parser.add_argument("--var", action="append", default=[], help="Extra key=value variable replacements.")
    parser.add_argument("--execute", action="store_true", help="Actually call the configured LLM endpoint.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--provider", choices=["openai", "gemini"], default=os.environ.get("GEM_LLM_PROVIDER", "openai"))
    args = parser.parse_args()

    variables: dict[str, str] = {}
    if args.text:
        variables["text"] = args.text.read_text(encoding="utf-8")
    for item in args.var:
        if "=" not in item:
            raise SystemExit(f"Invalid --var value: {item}. Expected key=value.")
        key, value = item.split("=", 1)
        variables[key] = value

    prompt = render_template(args.template.read_text(encoding="utf-8"), variables)
    if not args.execute:
        print(prompt)
        return
    print(call_model(prompt, provider=args.provider, temperature=args.temperature, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
