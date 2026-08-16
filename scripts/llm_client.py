#!/usr/bin/env python3
"""Small chat client shared by GEM request executors.

The default mode is dry-run: no network call is made. Set --execute and provide
provider credentials to call a local or remote LLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROVIDERS = ("openai", "responses", "gemini", "codex")


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
        if start >= 0:
            value, _end = json.JSONDecoder().raw_decode(text, start)
            return value
        raise


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def codex_prompt(messages: list[dict[str, str]], max_tokens: int) -> str:
    sections = [
        "Act as a JSON generation backend. Do not inspect files, run commands, or use tools.",
        (
            "Return only the requested JSON object with no markdown or commentary. "
            "Minify it: do not add indentation or insignificant whitespace."
        ),
        (
            "The caller requested a maximum of approximately "
            f"{max_tokens} output tokens; keep the JSON concise enough to fit."
        ),
    ]
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "user")).upper()
        sections.append(f"<message index=\"{index}\" role=\"{role}\">\n{message_text(message)}\n</message>")
    return "\n\n".join(sections)


def _nearest_project_settings(config: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    matches: list[tuple[int, dict[str, Any]]] = []
    for raw_path, settings in config.get("projects", {}).items():
        if not isinstance(raw_path, str) or not isinstance(settings, dict):
            continue
        try:
            project_dir.resolve().relative_to(Path(raw_path).expanduser().resolve())
        except ValueError:
            continue
        matches.append((len(Path(raw_path).parts), settings))
    return max(matches, key=lambda item: item[0])[1] if matches else {}


def _load_codex_config() -> tuple[Path | None, dict[str, Any], dict[str, Any]]:
    raw_path = os.environ.get("GEM_CODEX_CONFIG", "").strip()
    if not raw_path:
        return None, {}, {}
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"GEM_CODEX_CONFIG is not a file: {path}")
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Cannot read GEM_CODEX_CONFIG: {path}") from exc
    project_dir = Path(os.environ.get("GEM_CODEX_PROJECT_DIR", Path.cwd())).expanduser()
    return path, config, _nearest_project_settings(config, project_dir)


def _redact_secrets(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def call_codex(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    *,
    response_schema: dict[str, Any] | None = None,
    reasoning_effort_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Use the locally authenticated Codex CLI as a read-only GPT backend."""
    executable = os.environ.get("GEM_CODEX_COMMAND", "codex")
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"Codex executable not found: {executable!r}")
    timeout = int(os.environ.get("GEM_CODEX_TIMEOUT", "600"))
    config_path, config, project_settings = _load_codex_config()
    model = (
        os.environ.get("GEM_CODEX_MODEL", "").strip()
        or str(project_settings.get("model") or config.get("model") or "").strip()
    )
    provider = (
        os.environ.get("GEM_CODEX_PROVIDER", "").strip()
        or str(project_settings.get("model_provider") or config.get("model_provider") or "").strip()
    )
    reasoning_effort = reasoning_effort_override or (
        os.environ.get("GEM_CODEX_REASONING_EFFORT", "").strip()
        or str(
            project_settings.get("model_reasoning_effort")
            or config.get("model_reasoning_effort")
            or ""
        ).strip()
    )
    service_tier = (
        os.environ.get("GEM_CODEX_SERVICE_TIER", "").strip()
        or str(project_settings.get("service_tier") or config.get("service_tier") or "").strip()
    )
    ignore_user_config = os.environ.get("GEM_CODEX_IGNORE_USER_CONFIG", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if config_path and ignore_user_config:
        raise RuntimeError(
            "GEM_CODEX_CONFIG and GEM_CODEX_IGNORE_USER_CONFIG cannot be used together"
        )
    disable_mcp = os.environ.get("GEM_CODEX_DISABLE_MCP", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    child_env = os.environ.copy()
    secrets: list[str] = []
    if provider:
        provider_config = config.get("model_providers", {}).get(provider, {})
        if isinstance(provider_config, dict):
            env_key = provider_config.get("env_key")
            if isinstance(env_key, str) and env_key:
                embedded_secret = config.get(env_key)
                if isinstance(embedded_secret, str) and embedded_secret:
                    child_env[env_key] = embedded_secret
                    secrets.append(embedded_secret)
    prompt = codex_prompt(messages, max_tokens)
    # Codex may briefly finish a background plugin-cache write after the main
    # process exits. The directory contains only ephemeral read-only session
    # state, so a cleanup race must not discard an otherwise valid response.
    with tempfile.TemporaryDirectory(
        prefix="gem_codex_", ignore_cleanup_errors=True
    ) as temp_name:
        temp_dir = Path(temp_name)
        codex_home = temp_dir / "codex_home"
        codex_home.mkdir()
        if config_path:
            (codex_home / "config.toml").symlink_to(config_path)
            child_env["CODEX_HOME"] = str(codex_home)
        output_path = temp_dir / "last_message.json"
        command = [
            resolved,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--cd",
            str(temp_dir),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        if response_schema is not None:
            schema_path = temp_dir / "response_schema.json"
            schema_path.write_text(
                json.dumps(response_schema, ensure_ascii=False), encoding="utf-8"
            )
            command[2:2] = ["--output-schema", str(schema_path)]
        if model:
            command[2:2] = ["--model", model]
        if provider:
            command[2:2] = ["-c", f"model_provider={json.dumps(provider)}"]
        if reasoning_effort:
            command[2:2] = [
                "-c",
                f"model_reasoning_effort={json.dumps(reasoning_effort)}",
            ]
        if service_tier:
            command[2:2] = ["-c", f"service_tier={json.dumps(service_tier)}"]
        if disable_mcp:
            command[2:2] = ["-c", "mcp_servers={}"]
            for server_name in sorted(config.get("mcp_servers", {})):
                command[2:2] = [
                    "-c",
                    f"mcp_servers.{server_name}.enabled=false",
                ]
        if ignore_user_config:
            command[2:2] = ["--ignore-user-config"]
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                env=child_env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Codex request timed out after {timeout}s") from exc
        if completed.returncode != 0:
            detail = _redact_secrets(
                (completed.stderr or completed.stdout).strip(), secrets
            )
            raise RuntimeError(
                f"Codex request failed with exit {completed.returncode}: {detail[-1200:]}"
            )
        if not output_path.is_file():
            raise RuntimeError("Codex request completed without a final response file")
        raw = output_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError("Codex returned an empty final response")
    return raw, {
        "provider": "codex",
        "model": model or "current_codex_default",
        "model_provider": provider or "current_codex_default",
        "reasoning_effort": reasoning_effort or "current_codex_default",
        "service_tier": service_tier or "current_codex_default",
        "config_source": (
            "built_in_defaults"
            if ignore_user_config
            else "explicit_toml"
            if config_path
            else "current_user_config"
        ),
        "temperature_requested": temperature,
        "max_tokens_requested": max_tokens,
        "limits_are_advisory": True,
        "structured_output": response_schema is not None,
    }


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


def extract_responses_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    texts = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
    if not texts:
        raise RuntimeError("Responses API returned no output_text content")
    return "".join(texts)


def call_responses_compatible(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    *,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call an OpenAI-compatible Responses endpoint without SDK dependencies."""
    base_url = os.environ.get("GEM_RESPONSES_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("GEM_RESPONSES_API_KEY", "")
    model = os.environ.get("GEM_RESPONSES_MODEL", "")
    if not base_url or not api_key or not model:
        raise RuntimeError(
            "Set GEM_RESPONSES_BASE_URL, GEM_RESPONSES_API_KEY, and GEM_RESPONSES_MODEL."
        )
    input_messages = [
        {"role": message.get("role", "user"), "content": message_text(message)}
        for message in messages
        if message_text(message)
    ]
    payload: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    selected_effort = (
        reasoning_effort
        if reasoning_effort is not None
        else os.environ.get("GEM_RESPONSES_REASONING_EFFORT", "").strip()
    )
    if selected_effort:
        payload["reasoning"] = {"effort": selected_effort}
    service_tier = os.environ.get("GEM_RESPONSES_SERVICE_TIER", "").strip()
    if service_tier:
        payload["service_tier"] = service_tier
    if os.environ.get("GEM_RESPONSES_SEND_TEMPERATURE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        payload["temperature"] = temperature

    key_header = os.environ.get("GEM_RESPONSES_API_KEY_HEADER", "Authorization")
    key_value = api_key if key_header.casefold() != "authorization" else f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={key_header: key_value, "Content-Type": "application/json"},
        method="POST",
    )
    timeout = int(os.environ.get("GEM_RESPONSES_TIMEOUT", "300"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Responses API request failed with HTTP {exc.code}: {detail[:800]}"
        ) from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Responses API request failed: {exc}") from exc
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return extract_responses_text(body), {
        **usage,
        "provider": "responses",
        "model": model,
        "reasoning_effort": selected_effort or "provider_default",
        "service_tier": service_tier or "provider_default",
    }


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
    thinking_budget = os.environ.get("GEMINI_THINKING_BUDGET")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before using --provider gemini.")
    model_path = model if model.startswith("models/") else f"models/{model}"
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    if thinking_budget not in (None, ""):
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    payload = {
        "contents": gemini_contents(messages),
        "generationConfig": generation_config,
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
    usage = body.get("usageMetadata", {})
    candidates = body.get("candidates") or []
    if candidates and candidates[0].get("finishReason"):
        usage = {**usage, "finishReason": candidates[0]["finishReason"]}
    return extract_gemini_text(body), usage


def call_chat(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    provider: str,
    *,
    reasoning_effort: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if provider == "gemini":
        return call_gemini(messages, max_tokens=max_tokens, temperature=temperature)
    if provider == "codex":
        return call_codex(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            reasoning_effort_override=reasoning_effort,
        )
    if provider == "responses":
        return call_responses_compatible(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if provider != "openai":
        raise RuntimeError(f"Unsupported provider: {provider!r}")
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
    parser.add_argument("--provider", choices=PROVIDERS, default=os.environ.get("GEM_LLM_PROVIDER", "openai"))
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
