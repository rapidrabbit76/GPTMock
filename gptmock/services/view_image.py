from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

VIEW_IMAGE_TOOL_NAME = "view_image"
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024

VIEW_IMAGE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "name": VIEW_IMAGE_TOOL_NAME,
    "description": (
        "View a local image from the filesystem. Use only when the user provided "
        "a local image path that GPTMock is allowed to read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local filesystem path to an image file.",
            },
            "detail": {
                "type": "string",
                "description": (
                    "Optional detail override. The only supported value is `original`; "
                    "omit this field for default behavior."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}


def normalize_view_image_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Normalize shorthand or chat-style view_image tool declarations."""

    normalized: list[dict[str, Any]] = []
    enabled = False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if _is_view_image_tool_spec(tool):
            normalized.append(dict(VIEW_IMAGE_TOOL_SPEC))
            enabled = True
        else:
            normalized.append(tool)
    return normalized, enabled


def is_view_image_tool_call(call: dict[str, Any]) -> bool:
    return (
        isinstance(call, dict)
        and call.get("type") == "function_call"
        and call.get("name") == VIEW_IMAGE_TOOL_NAME
    )


def execute_view_image(arguments: Any) -> str | list[dict[str, Any]]:
    """Execute a view_image tool call and return a Responses function output payload."""

    args = _parse_arguments(arguments)
    path_arg = args.get("path")
    if not isinstance(path_arg, str) or not path_arg.strip():
        return "view_image requires a non-empty `path` string"

    detail = args.get("detail")
    if detail is not None and detail != "original":
        return (
            "view_image.detail only supports `original`; omit `detail` for "
            f"default behavior, got `{detail}`"
        )

    path = _resolve_image_path(path_arg)
    if path is None:
        return f"view_image path is outside the allowed roots: {path_arg}"
    if not path.is_file():
        return f"view_image path is not a file: {path}"

    max_bytes = _max_image_bytes()
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"view_image failed to stat file: {exc}"
    if size > max_bytes:
        return f"view_image file is too large: {size} bytes exceeds limit {max_bytes}"

    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"view_image failed to read file: {exc}"

    mime_type = _detect_image_mime_type(path, data)
    if mime_type is None:
        return f"view_image only supports image files with PNG, JPEG, GIF, or WebP content: {path}"

    item: dict[str, Any] = {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}",
    }
    if detail == "original":
        item["detail"] = "original"
    return [item]


def _is_view_image_tool_spec(tool: dict[str, Any]) -> bool:
    if tool.get("type") == VIEW_IMAGE_TOOL_NAME:
        return True
    if tool.get("type") == "function" and tool.get("name") == VIEW_IMAGE_TOOL_NAME:
        return True
    nested = tool.get("function")
    return (
        tool.get("type") == "function"
        and isinstance(nested, dict)
        and nested.get("name") == VIEW_IMAGE_TOOL_NAME
    )


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_image_path(path_arg: str) -> Path | None:
    raw = Path(path_arg).expanduser()
    path = raw if raw.is_absolute() else Path.cwd() / raw
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return None

    if _env_truthy("GPTMOCK_VIEW_IMAGE_ALLOW_ANY_PATH"):
        return resolved

    roots = _allowed_roots()
    if any(_is_relative_to(resolved, root) for root in roots):
        return resolved
    return None


def _allowed_roots() -> list[Path]:
    raw_roots = os.getenv("GPTMOCK_VIEW_IMAGE_ROOTS")
    if raw_roots:
        roots = [part for part in raw_roots.split(os.pathsep) if part.strip()]
    else:
        roots = [str(Path.cwd())]

    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(Path(root).expanduser().resolve(strict=False))
        except OSError:
            continue
    return resolved or [Path.cwd().resolve(strict=False)]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _max_image_bytes() -> int:
    raw = os.getenv("GPTMOCK_VIEW_IMAGE_MAX_BYTES")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = _DEFAULT_MAX_IMAGE_BYTES
        return max(1, value)
    return _DEFAULT_MAX_IMAGE_BYTES


def _detect_image_mime_type(path: Path, data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    guessed, _ = mimetypes.guess_type(path.name)
    if guessed in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        return guessed
    return None


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
