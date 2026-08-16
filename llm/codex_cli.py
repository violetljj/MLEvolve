"""Codex CLI backend for ChatGPT-authenticated local MLEvolve runs.

This adapter preserves MLEvolve's LLM-facing interface while delegating each
request to ``codex exec``.  The command is supplied as a JSON array through
``MLEVOLVE_CODEX_COMMAND`` so Windows installations do not have to rely on a
WindowsApps shim or shell parsing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from config import Config
from .gemini import FunctionSpec, OutputType, compile_prompt_to_md

logger = logging.getLogger("MLEvolve")


def _strict_schema(schema: dict) -> dict:
    """Add the closed-object markers required by Codex structured output."""
    result = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                properties = node.get("properties", {})
                required = set(node.get("required", []))
                for name, prop in properties.items():
                    if name not in required and isinstance(prop, dict):
                        prop_type = prop.get("type")
                        if isinstance(prop_type, str):
                            prop["type"] = [prop_type, "null"]
                        elif isinstance(prop_type, list) and "null" not in prop_type:
                            prop["type"] = [*prop_type, "null"]
                        else:
                            properties[name] = {"anyOf": [prop, {"type": "null"}]}
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def _command() -> list[str]:
    raw = os.getenv("MLEVOLVE_CODEX_COMMAND", "")
    if not raw:
        return ["codex"]
    command = json.loads(raw)
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError("MLEVOLVE_CODEX_COMMAND must be a non-empty JSON string array")
    return command


def _model_name(model: str) -> str:
    if (model or "").lower().startswith("codex:"):
        return model.split(":", 1)[1]
    return model


def _combine_prompt(system_message: str | None, user_message: str | None) -> str:
    parts = []
    if system_message:
        parts.append(f"# System instructions\n\n{system_message.strip()}")
    if user_message:
        parts.append(f"# User request\n\n{user_message.strip()}")
    if not parts:
        raise ValueError("Either system_message or user_message must be provided")
    return "\n\n".join(parts)


def _run(prompt: str, model: str, json_schema: dict | None = None) -> tuple[str, float]:
    output_path: Path | None = None
    schema_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as output_file:
            output_path = Path(output_file.name)

        args = _command() + [
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--color",
            "never",
            "-m",
            _model_name(model),
            "-c",
            f'model_reasoning_effort="{os.getenv("MLEVOLVE_CODEX_REASONING_EFFORT", "medium")}"',
            "-o",
            str(output_path),
        ]

        if json_schema is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as schema_file:
                json.dump(_strict_schema(json_schema), schema_file, ensure_ascii=False)
                schema_path = Path(schema_file.name)
            args.extend(["--output-schema", str(schema_path)])

        args.append("-")
        logger.info("Querying Codex CLI with model: %s", _model_name(model))
        started = time.time()
        completed = subprocess.run(
            args,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=int(os.getenv("MLEVOLVE_CODEX_TIMEOUT", "1800")),
            check=False,
        )
        elapsed = time.time() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"codex exec failed with exit code {completed.returncode}: {completed.stderr[-4000:]}"
            )
        output = output_path.read_text(encoding="utf-8").strip()
        if not output:
            raise ValueError("codex exec returned an empty final response")
        logger.info("Codex CLI response: %s", output, extra={"verbose": True})
        return output, elapsed
    finally:
        for path in (output_path, schema_path):
            if path is not None:
                path.unlink(missing_ok=True)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    del cfg
    model = model_kwargs.get("model", "codex:gpt-5.6-sol")
    prompt = _combine_prompt(system_message, user_message)
    if func_spec is not None:
        prompt += (
            f"\n\nReturn only arguments for the `{func_spec.name}` result. "
            "Your final response must conform exactly to the supplied JSON Schema."
        )
    text, elapsed = _run(prompt, model, func_spec.json_schema if func_spec else None)
    output: OutputType = json.loads(text) if func_spec else text
    return output, elapsed, 0, 0, {"model": _model_name(model), "created": int(time.time())}


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 3,
    retry_delay: float = 3,
) -> str:
    del temperature, max_tokens, stop_tokens
    if not isinstance(prompt, str):
        prompt = compile_prompt_to_md(prompt)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            text, _ = _run(prompt, cfg.agent.code.model, json_schema)
            return text
        except Exception as exc:
            last_error = exc
            logger.warning("Codex CLI generation failed (%s/%s): %s", attempt + 1, max_retries, exc)
            if attempt + 1 < max_retries:
                time.sleep(retry_delay)
    assert last_error is not None
    raise last_error
