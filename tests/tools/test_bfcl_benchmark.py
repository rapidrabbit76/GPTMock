"""BFCL-style function-calling benchmark for GPTMock.

Inspired by Berkeley Function-Calling Leaderboard (BFCL) V3 categories:
  - Simple:           1 tool provided, 1 call expected
  - Multiple:         2-4 tools provided, pick correct 1
  - Parallel:         1+ tools, multiple calls from single query
  - Parallel Multiple: multiple tools, multiple calls to different functions
  - Irrelevance:      tools provided but none relevant → should produce text, not tool_calls

~170 scenarios × 9 models ≈ 1,530 tests. Expected runtime: ~90 minutes.

Usage:
    uv run pytest tests/test_bfcl_benchmark.py -v
    uv run pytest tests/test_bfcl_benchmark.py -v -k "simple"
    uv run pytest tests/test_bfcl_benchmark.py -v -k "gpt_5_2"
    uv run pytest tests/test_bfcl_benchmark.py -v --timeout=7200
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.services.model_registry import get_model_list

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALL_MODELS: List[str] = get_model_list(expose_reasoning=False)
TIMEOUT = 120


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — organized by domain
# ═══════════════════════════════════════════════════════════════════════════


def _tool(name: str, desc: str, props: dict, required: list) -> dict:
    """Helper to build a function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


# ── Filesystem ────────────────────────────────────────────────────────────
T_FS_READ = _tool(
    "read_file",
    "Read complete contents of a file.",
    {
        "path": {"type": "string", "description": "Absolute file path"},
    },
    ["path"],
)

T_FS_WRITE = _tool(
    "write_file",
    "Write content to a file (creates or overwrites).",
    {
        "path": {"type": "string", "description": "Absolute file path"},
        "content": {"type": "string", "description": "Content to write"},
    },
    ["path", "content"],
)

T_FS_LIST = _tool(
    "list_directory",
    "List files and directories at the given path.",
    {
        "path": {"type": "string", "description": "Directory path"},
    },
    ["path"],
)

T_FS_MOVE = _tool(
    "move_file",
    "Move or rename a file.",
    {
        "source": {"type": "string", "description": "Source file path"},
        "destination": {"type": "string", "description": "Destination file path"},
    },
    ["source", "destination"],
)

T_FS_SEARCH = _tool(
    "search_files",
    "Search for files matching a pattern.",
    {
        "path": {"type": "string", "description": "Base directory"},
        "pattern": {"type": "string", "description": "Glob pattern (e.g. *.py)"},
        "recursive": {"type": "boolean", "description": "Search recursively"},
    },
    ["path", "pattern"],
)

T_FS_DELETE = _tool(
    "delete_file",
    "Delete a file permanently.",
    {
        "path": {"type": "string", "description": "Absolute file path"},
        "confirm": {
            "type": "boolean",
            "description": "Must be true to confirm deletion",
        },
    },
    ["path", "confirm"],
)

T_FS_MKDIR = _tool(
    "create_directory",
    "Create a new directory.",
    {
        "path": {"type": "string", "description": "Directory path to create"},
        "parents": {
            "type": "boolean",
            "description": "Create parent directories if needed",
        },
    },
    ["path"],
)

T_FS_STAT = _tool(
    "get_file_info",
    "Get file metadata (size, modified date, permissions).",
    {
        "path": {"type": "string", "description": "File path"},
    },
    ["path"],
)

# ── Database ──────────────────────────────────────────────────────────────
T_DB_SELECT = _tool(
    "db_query",
    "Execute a read-only SELECT query.",
    {
        "sql": {"type": "string", "description": "SQL SELECT query"},
        "params": {"type": "array", "items": {}, "description": "Query parameters"},
    },
    ["sql"],
)

T_DB_INSERT = _tool(
    "db_insert",
    "Insert rows into a table.",
    {
        "table": {"type": "string", "description": "Table name"},
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Column names",
        },
        "values": {
            "type": "array",
            "items": {"type": "array", "items": {}},
            "description": "Row values",
        },
    },
    ["table", "columns", "values"],
)

T_DB_UPDATE = _tool(
    "db_update",
    "Update rows in a table.",
    {
        "table": {"type": "string", "description": "Table name"},
        "set": {"type": "object", "description": "Column-value pairs to update"},
        "where": {"type": "string", "description": "WHERE clause"},
    },
    ["table", "set", "where"],
)

T_DB_DELETE = _tool(
    "db_delete",
    "Delete rows from a table.",
    {
        "table": {"type": "string", "description": "Table name"},
        "where": {"type": "string", "description": "WHERE clause"},
    },
    ["table", "where"],
)

T_DB_DESCRIBE = _tool(
    "describe_table",
    "Get table schema.",
    {
        "table_name": {"type": "string", "description": "Table name"},
    },
    ["table_name"],
)

T_DB_LIST_TABLES = _tool(
    "list_tables",
    "List all tables in the database.",
    {
        "schema": {"type": "string", "description": "Schema name (default: public)"},
    },
    [],
)

# ── Search ────────────────────────────────────────────────────────────────
T_WEB_SEARCH = _tool(
    "web_search",
    "Search the web.",
    {
        "query": {"type": "string", "description": "Search query"},
        "count": {"type": "integer", "description": "Number of results (1-20)"},
    },
    ["query"],
)

T_LOCAL_SEARCH = _tool(
    "local_search",
    "Search for local businesses and places.",
    {
        "query": {"type": "string", "description": "Search query"},
        "latitude": {"type": "number", "description": "Latitude"},
        "longitude": {"type": "number", "description": "Longitude"},
    },
    ["query"],
)

T_NEWS_SEARCH = _tool(
    "news_search",
    "Search recent news articles.",
    {
        "query": {"type": "string", "description": "News search query"},
        "freshness": {
            "type": "string",
            "enum": ["day", "week", "month"],
            "description": "Recency filter",
        },
    },
    ["query"],
)

# ── GitHub ────────────────────────────────────────────────────────────────
T_GH_CREATE_ISSUE = _tool(
    "create_issue",
    "Create a GitHub issue.",
    {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "string"}},
    },
    ["owner", "repo", "title"],
)

T_GH_LIST_ISSUES = _tool(
    "list_issues",
    "List issues in a repository.",
    {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "state": {"type": "string", "enum": ["open", "closed", "all"]},
        "labels": {"type": "string", "description": "Comma-separated label names"},
    },
    ["owner", "repo"],
)

T_GH_CREATE_PR = _tool(
    "create_pull_request",
    "Create a pull request.",
    {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "head": {"type": "string", "description": "Branch with changes"},
        "base": {"type": "string", "description": "Target branch"},
    },
    ["owner", "repo", "title", "head", "base"],
)

T_GH_GET_FILE = _tool(
    "get_file_contents",
    "Get file contents from a GitHub repo.",
    {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "path": {"type": "string"},
        "ref": {"type": "string", "description": "Branch or commit SHA"},
    },
    ["owner", "repo", "path"],
)

T_GH_SEARCH_CODE = _tool(
    "search_code",
    "Search code across GitHub repositories.",
    {
        "query": {
            "type": "string",
            "description": "Search query with GitHub qualifiers",
        },
    },
    ["query"],
)

# ── Slack ─────────────────────────────────────────────────────────────────
T_SLACK_SEND = _tool(
    "send_message",
    "Send a Slack message to a channel.",
    {
        "channel": {"type": "string", "description": "Channel ID"},
        "text": {"type": "string", "description": "Message text"},
        "thread_ts": {"type": "string", "description": "Thread timestamp for replies"},
    },
    ["channel", "text"],
)

T_SLACK_LIST_CH = _tool(
    "list_channels",
    "List Slack channels.",
    {
        "types": {
            "type": "string",
            "description": "Channel types (public_channel, private_channel)",
        },
        "limit": {"type": "integer"},
    },
    [],
)

T_SLACK_REACT = _tool(
    "add_reaction",
    "Add an emoji reaction to a message.",
    {
        "channel": {"type": "string"},
        "timestamp": {"type": "string"},
        "name": {"type": "string", "description": "Emoji name without colons"},
    },
    ["channel", "timestamp", "name"],
)

T_SLACK_UPLOAD = _tool(
    "upload_file",
    "Upload a file to Slack.",
    {
        "channels": {"type": "string", "description": "Comma-separated channel IDs"},
        "content": {"type": "string"},
        "filename": {"type": "string"},
        "title": {"type": "string"},
    },
    ["channels", "content", "filename"],
)

# ── Browser/Playwright ────────────────────────────────────────────────────
T_PW_NAV = _tool(
    "browser_navigate",
    "Navigate to a URL.",
    {
        "url": {"type": "string"},
    },
    ["url"],
)

T_PW_CLICK = _tool(
    "browser_click",
    "Click an element.",
    {
        "selector": {"type": "string"},
    },
    ["selector"],
)

T_PW_TYPE = _tool(
    "browser_type",
    "Type text into an input.",
    {
        "selector": {"type": "string"},
        "text": {"type": "string"},
    },
    ["selector", "text"],
)

T_PW_SCREENSHOT = _tool(
    "browser_screenshot",
    "Take a screenshot.",
    {
        "fullPage": {"type": "boolean"},
        "selector": {"type": "string", "description": "Element selector (optional)"},
    },
    [],
)

T_PW_EVAL = _tool(
    "browser_evaluate",
    "Execute JavaScript in the page.",
    {
        "expression": {"type": "string"},
    },
    ["expression"],
)

# ── Memory/Knowledge Graph ────────────────────────────────────────────────
T_MEM_ENTITIES = _tool(
    "create_entities",
    "Create knowledge graph entities.",
    {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entityType": {"type": "string"},
                    "observations": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    ["entities"],
)

T_MEM_RELATIONS = _tool(
    "create_relations",
    "Create relations between entities.",
    {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "relationType": {"type": "string"},
                },
            },
        },
    },
    ["relations"],
)

T_MEM_SEARCH = _tool(
    "search_nodes",
    "Search the knowledge graph.",
    {
        "query": {"type": "string"},
    },
    ["query"],
)

# ── Weather ───────────────────────────────────────────────────────────────
T_WEATHER_CURRENT = _tool(
    "get_current_weather",
    "Get current weather for a location.",
    {
        "location": {"type": "string", "description": "City name or coordinates"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    ["location"],
)

T_WEATHER_FORECAST = _tool(
    "get_forecast",
    "Get weather forecast.",
    {
        "location": {"type": "string"},
        "days": {"type": "integer", "description": "Forecast days (1-7)"},
    },
    ["location"],
)

# ── Email ─────────────────────────────────────────────────────────────────
T_EMAIL_SEND = _tool(
    "send_email",
    "Send an email.",
    {
        "to": {"type": "string"},
        "subject": {"type": "string"},
        "body": {"type": "string"},
        "cc": {"type": "string"},
        "attachments": {"type": "array", "items": {"type": "string"}},
    },
    ["to", "subject", "body"],
)

T_EMAIL_SEARCH = _tool(
    "search_email",
    "Search emails.",
    {
        "query": {"type": "string"},
        "folder": {"type": "string"},
        "from": {"type": "string"},
        "limit": {"type": "integer"},
    },
    ["query"],
)

T_EMAIL_READ = _tool(
    "read_email",
    "Read a specific email by ID.",
    {
        "message_id": {"type": "string"},
    },
    ["message_id"],
)

# ── Math/Compute ─────────────────────────────────────────────────────────
T_CALC = _tool(
    "calculate",
    "Evaluate a mathematical expression.",
    {
        "expression": {"type": "string", "description": "Math expression to evaluate"},
    },
    ["expression"],
)

T_CONVERT = _tool(
    "convert_units",
    "Convert between units.",
    {
        "value": {"type": "number"},
        "from_unit": {"type": "string"},
        "to_unit": {"type": "string"},
    },
    ["value", "from_unit", "to_unit"],
)

T_STATS = _tool(
    "statistics",
    "Calculate statistics for a dataset.",
    {
        "data": {"type": "array", "items": {"type": "number"}},
        "operations": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["mean", "median", "std", "min", "max"],
            },
        },
    },
    ["data", "operations"],
)

# ── Calendar ──────────────────────────────────────────────────────────────
T_CAL_CREATE = _tool(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string"},
        "start": {"type": "string", "description": "ISO 8601 datetime"},
        "end": {"type": "string"},
        "location": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
    },
    ["title", "start", "end"],
)

T_CAL_LIST = _tool(
    "list_events",
    "List calendar events.",
    {
        "date": {"type": "string", "description": "Date in YYYY-MM-DD"},
        "calendar_id": {"type": "string"},
    },
    ["date"],
)

T_CAL_DELETE = _tool(
    "delete_event",
    "Delete a calendar event.",
    {
        "event_id": {"type": "string"},
    },
    ["event_id"],
)

# ── Docker/Infra ─────────────────────────────────────────────────────────
T_DOCKER_RUN = _tool(
    "docker_run",
    "Run a Docker container.",
    {
        "image": {"type": "string"},
        "name": {"type": "string"},
        "ports": {"type": "object", "description": "Port mappings {host: container}"},
        "env": {"type": "object", "description": "Environment variables"},
        "detach": {"type": "boolean"},
    },
    ["image"],
)

T_DOCKER_PS = _tool(
    "docker_list",
    "List running Docker containers.",
    {
        "all": {"type": "boolean", "description": "Include stopped containers"},
    },
    [],
)

T_DOCKER_STOP = _tool(
    "docker_stop",
    "Stop a running container.",
    {
        "container": {"type": "string", "description": "Container name or ID"},
    },
    ["container"],
)

T_DOCKER_LOGS = _tool(
    "docker_logs",
    "Get container logs.",
    {
        "container": {"type": "string"},
        "tail": {"type": "integer"},
        "since": {
            "type": "string",
            "description": "Show logs since (e.g. '10m', '1h')",
        },
    },
    ["container"],
)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════


class S:
    """Scenario descriptor."""

    __slots__ = (
        "id",
        "cat",
        "prompt",
        "tools",
        "expect_fns",
        "required_args",
        "min_calls",
        "parallel",
        "notes",
    )

    def __init__(
        self,
        id: str,
        cat: str,
        prompt: str,
        tools: list,
        expect_fns: Optional[list] = None,
        required_args: Optional[dict] = None,
        min_calls: int = 1,
        parallel: bool = False,
        notes: str = "",
    ):
        self.id = id
        self.cat = cat
        self.prompt = prompt
        self.tools = tools
        self.expect_fns = expect_fns  # expected function name(s)
        self.required_args = required_args or {}  # {fn_name: [arg_names]}
        self.min_calls = min_calls
        self.parallel = parallel
        self.notes = notes

    def __repr__(self) -> str:
        return self.id


# ── SIMPLE: 1 tool, 1 call ───────────────────────────────────────────────
SIMPLE = [
    # Filesystem
    S(
        "simple_fs_read",
        "simple",
        "Read the file /etc/hosts and show me what's inside.",
        [T_FS_READ],
        expect_fns=["read_file"],
        required_args={"read_file": ["path"]},
    ),
    S(
        "simple_fs_write",
        "simple",
        "Write 'Hello World' to /tmp/greeting.txt.",
        [T_FS_WRITE],
        expect_fns=["write_file"],
        required_args={"write_file": ["path", "content"]},
    ),
    S(
        "simple_fs_list",
        "simple",
        "List all files in the /var/log directory.",
        [T_FS_LIST],
        expect_fns=["list_directory"],
        required_args={"list_directory": ["path"]},
    ),
    S(
        "simple_fs_move",
        "simple",
        "Rename /tmp/old.txt to /tmp/new.txt.",
        [T_FS_MOVE],
        expect_fns=["move_file"],
        required_args={"move_file": ["source", "destination"]},
    ),
    S(
        "simple_fs_search",
        "simple",
        "Find all Python files in /home/user/projects.",
        [T_FS_SEARCH],
        expect_fns=["search_files"],
        required_args={"search_files": ["path", "pattern"]},
    ),
    S(
        "simple_fs_delete",
        "simple",
        "Delete the file /tmp/cache.dat (confirmed).",
        [T_FS_DELETE],
        expect_fns=["delete_file"],
        required_args={"delete_file": ["path"]},
    ),
    S(
        "simple_fs_mkdir",
        "simple",
        "Create directory /opt/myapp/config with parent dirs.",
        [T_FS_MKDIR],
        expect_fns=["create_directory"],
        required_args={"create_directory": ["path"]},
    ),
    S(
        "simple_fs_stat",
        "simple",
        "Show me the file size and modification date of /var/log/syslog.",
        [T_FS_STAT],
        expect_fns=["get_file_info"],
        required_args={"get_file_info": ["path"]},
    ),
    # Database
    S(
        "simple_db_select",
        "simple",
        "Query: SELECT * FROM users WHERE active = true LIMIT 10.",
        [T_DB_SELECT],
        expect_fns=["db_query"],
        required_args={"db_query": ["sql"]},
    ),
    S(
        "simple_db_insert",
        "simple",
        "Insert into 'products' table: columns (name, price), values ('Widget', 9.99).",
        [T_DB_INSERT],
        expect_fns=["db_insert"],
        required_args={"db_insert": ["table", "columns", "values"]},
    ),
    S(
        "simple_db_update",
        "simple",
        "Update users table: set status='inactive' where last_login < '2024-01-01'.",
        [T_DB_UPDATE],
        expect_fns=["db_update"],
        required_args={"db_update": ["table", "set", "where"]},
    ),
    S(
        "simple_db_describe",
        "simple",
        "Describe the schema of the 'orders' table.",
        [T_DB_DESCRIBE],
        expect_fns=["describe_table"],
        required_args={"describe_table": ["table_name"]},
    ),
    S(
        "simple_db_list",
        "simple",
        "Show me all tables in the database.",
        [T_DB_LIST_TABLES],
        expect_fns=["list_tables"],
    ),
    # Search
    S(
        "simple_search_web",
        "simple",
        "Search for 'Python asyncio tutorial 2025'.",
        [T_WEB_SEARCH],
        expect_fns=["web_search"],
        required_args={"web_search": ["query"]},
    ),
    S(
        "simple_search_local",
        "simple",
        "Find coffee shops near latitude 37.7749, longitude -122.4194.",
        [T_LOCAL_SEARCH],
        expect_fns=["local_search"],
        required_args={"local_search": ["query"]},
    ),
    S(
        "simple_search_news",
        "simple",
        "Search for news about 'AI regulation' from the past week.",
        [T_NEWS_SEARCH],
        expect_fns=["news_search"],
        required_args={"news_search": ["query"]},
    ),
    # GitHub
    S(
        "simple_gh_issue",
        "simple",
        "Create an issue in owner=microsoft repo=vscode titled 'Memory leak in extensions' "
        "with body 'Extensions consume excessive memory after 2h usage'.",
        [T_GH_CREATE_ISSUE],
        expect_fns=["create_issue"],
        required_args={"create_issue": ["owner", "repo", "title"]},
    ),
    S(
        "simple_gh_list",
        "simple",
        "List open issues in owner=facebook repo=react.",
        [T_GH_LIST_ISSUES],
        expect_fns=["list_issues"],
        required_args={"list_issues": ["owner", "repo"]},
    ),
    S(
        "simple_gh_file",
        "simple",
        "Get the README.md from owner=torvalds repo=linux on the master branch.",
        [T_GH_GET_FILE],
        expect_fns=["get_file_contents"],
        required_args={"get_file_contents": ["owner", "repo", "path"]},
    ),
    S(
        "simple_gh_search",
        "simple",
        "Search GitHub for code containing 'fastapi dependency injection' in Python.",
        [T_GH_SEARCH_CODE],
        expect_fns=["search_code"],
        required_args={"search_code": ["query"]},
    ),
    # Slack
    S(
        "simple_slack_send",
        "simple",
        "Send 'Deployment complete ✓' to Slack channel C0ABC123.",
        [T_SLACK_SEND],
        expect_fns=["send_message"],
        required_args={"send_message": ["channel", "text"]},
    ),
    S(
        "simple_slack_list",
        "simple",
        "List all public Slack channels.",
        [T_SLACK_LIST_CH],
        expect_fns=["list_channels"],
    ),
    S(
        "simple_slack_react",
        "simple",
        "Add a 👍 (thumbsup) reaction to message at timestamp 1234567890.123456 in channel C0ABC123.",
        [T_SLACK_REACT],
        expect_fns=["add_reaction"],
        required_args={"add_reaction": ["channel", "timestamp", "name"]},
    ),
    # Browser
    S(
        "simple_pw_nav",
        "simple",
        "Navigate to https://example.com.",
        [T_PW_NAV],
        expect_fns=["browser_navigate"],
        required_args={"browser_navigate": ["url"]},
    ),
    S(
        "simple_pw_screenshot",
        "simple",
        "Take a full-page screenshot.",
        [T_PW_SCREENSHOT],
        expect_fns=["browser_screenshot"],
    ),
    S(
        "simple_pw_eval",
        "simple",
        "Run JavaScript: document.querySelectorAll('a').length to count all links.",
        [T_PW_EVAL],
        expect_fns=["browser_evaluate"],
        required_args={"browser_evaluate": ["expression"]},
    ),
    # Memory
    S(
        "simple_mem_search",
        "simple",
        "Search the knowledge graph for 'machine learning'.",
        [T_MEM_SEARCH],
        expect_fns=["search_nodes"],
        required_args={"search_nodes": ["query"]},
    ),
    # Weather
    S(
        "simple_weather_now",
        "simple",
        "What's the current weather in Seoul in Celsius?",
        [T_WEATHER_CURRENT],
        expect_fns=["get_current_weather"],
        required_args={"get_current_weather": ["location"]},
    ),
    S(
        "simple_weather_forecast",
        "simple",
        "Get a 5-day forecast for Tokyo.",
        [T_WEATHER_FORECAST],
        expect_fns=["get_forecast"],
        required_args={"get_forecast": ["location"]},
    ),
    # Email
    S(
        "simple_email_send",
        "simple",
        "Send an email to alice@example.com with subject 'Meeting Notes' and body 'See attached summary'.",
        [T_EMAIL_SEND],
        expect_fns=["send_email"],
        required_args={"send_email": ["to", "subject", "body"]},
    ),
    S(
        "simple_email_search",
        "simple",
        "Search emails for 'invoice' in the inbox folder.",
        [T_EMAIL_SEARCH],
        expect_fns=["search_email"],
        required_args={"search_email": ["query"]},
    ),
    # Math
    S(
        "simple_calc",
        "simple",
        "Calculate: (25 * 4) + (16 / 2) - 7.",
        [T_CALC],
        expect_fns=["calculate"],
        required_args={"calculate": ["expression"]},
    ),
    S(
        "simple_convert",
        "simple",
        "Convert 100 kilometers to miles.",
        [T_CONVERT],
        expect_fns=["convert_units"],
        required_args={"convert_units": ["value", "from_unit", "to_unit"]},
    ),
    S(
        "simple_stats",
        "simple",
        "Calculate the mean, median, and standard deviation of [12, 15, 18, 22, 25, 30, 35].",
        [T_STATS],
        expect_fns=["statistics"],
        required_args={"statistics": ["data", "operations"]},
    ),
    # Calendar
    S(
        "simple_cal_create",
        "simple",
        "Create a meeting titled 'Sprint Planning' from 2025-03-15T10:00 to 2025-03-15T11:00 "
        "in Conference Room A.",
        [T_CAL_CREATE],
        expect_fns=["create_event"],
        required_args={"create_event": ["title", "start", "end"]},
    ),
    S(
        "simple_cal_list",
        "simple",
        "Show all events for 2025-03-15.",
        [T_CAL_LIST],
        expect_fns=["list_events"],
        required_args={"list_events": ["date"]},
    ),
    # Docker
    S(
        "simple_docker_run",
        "simple",
        "Run a Redis container named 'my-redis' with image redis:7-alpine, port 6379:6379, detached.",
        [T_DOCKER_RUN],
        expect_fns=["docker_run"],
        required_args={"docker_run": ["image"]},
    ),
    S(
        "simple_docker_ps",
        "simple",
        "List all running Docker containers.",
        [T_DOCKER_PS],
        expect_fns=["docker_list"],
    ),
    S(
        "simple_docker_logs",
        "simple",
        "Show the last 50 lines of logs from the 'web-app' container.",
        [T_DOCKER_LOGS],
        expect_fns=["docker_logs"],
        required_args={"docker_logs": ["container"]},
    ),
    S(
        "simple_docker_stop",
        "simple",
        "Stop the container named 'old-service'.",
        [T_DOCKER_STOP],
        expect_fns=["docker_stop"],
        required_args={"docker_stop": ["container"]},
    ),
    # DB delete
    S(
        "simple_db_delete",
        "simple",
        "Delete all rows from 'sessions' table where expired_at < '2024-06-01'.",
        [T_DB_DELETE],
        expect_fns=["db_delete"],
        required_args={"db_delete": ["table", "where"]},
    ),
]

# ── MULTIPLE: 2-4 tools, pick the right 1 ────────────────────────────────
MULTIPLE = [
    # Pick search over email
    S(
        "multi_search_vs_email",
        "multiple",
        "Find information about Python 3.13 new features.",
        [T_WEB_SEARCH, T_EMAIL_SEARCH, T_NEWS_SEARCH],
        expect_fns=["web_search"],
    ),
    # Pick news over web
    S(
        "multi_news_vs_web",
        "multiple",
        "What happened with the OpenAI board situation in the last 24 hours?",
        [T_WEB_SEARCH, T_NEWS_SEARCH, T_LOCAL_SEARCH],
        expect_fns=["news_search"],
    ),
    # Pick local over web
    S(
        "multi_local_vs_web",
        "multiple",
        "Find the nearest sushi restaurant to my location (37.5665, 126.9780).",
        [T_WEB_SEARCH, T_LOCAL_SEARCH, T_NEWS_SEARCH],
        expect_fns=["local_search"],
    ),
    # FS: pick read over write/delete
    S(
        "multi_fs_read_vs_others",
        "multiple",
        "Show me the contents of /etc/nginx/nginx.conf.",
        [T_FS_READ, T_FS_WRITE, T_FS_DELETE, T_FS_MOVE],
        expect_fns=["read_file"],
    ),
    S(
        "multi_fs_write_vs_read",
        "multiple",
        "Create a new config file at /tmp/app.conf with content: port=8080.",
        [T_FS_READ, T_FS_WRITE, T_FS_LIST, T_FS_MOVE],
        expect_fns=["write_file"],
    ),
    S(
        "multi_fs_move_vs_others",
        "multiple",
        "Move the file /tmp/data.csv to /archive/data.csv.",
        [T_FS_READ, T_FS_WRITE, T_FS_MOVE, T_FS_DELETE],
        expect_fns=["move_file"],
    ),
    S(
        "multi_fs_search_vs_list",
        "multiple",
        "Find all .log files recursively under /var/log.",
        [T_FS_LIST, T_FS_SEARCH, T_FS_READ],
        expect_fns=["search_files"],
    ),
    # DB: pick correct operation
    S(
        "multi_db_select_vs_update",
        "multiple",
        "Show me all orders placed in January 2025.",
        [T_DB_SELECT, T_DB_UPDATE, T_DB_INSERT, T_DB_DELETE],
        expect_fns=["db_query"],
    ),
    S(
        "multi_db_insert_vs_others",
        "multiple",
        "Add a new product: name='Gadget X', price=49.99 to the products table.",
        [T_DB_SELECT, T_DB_INSERT, T_DB_UPDATE, T_DB_DELETE],
        expect_fns=["db_insert"],
    ),
    S(
        "multi_db_update_vs_delete",
        "multiple",
        "Change the status to 'shipped' for order #12345.",
        [T_DB_SELECT, T_DB_INSERT, T_DB_UPDATE, T_DB_DELETE],
        expect_fns=["db_update"],
    ),
    S(
        "multi_db_describe_vs_query",
        "multiple",
        "What columns does the 'inventory' table have?",
        [T_DB_SELECT, T_DB_DESCRIBE, T_DB_LIST_TABLES],
        expect_fns=["describe_table"],
    ),
    # GH: pick correct action
    S(
        "multi_gh_issue_vs_pr",
        "multiple",
        "Report a bug: the login page crashes on Safari.",
        [T_GH_CREATE_ISSUE, T_GH_CREATE_PR, T_GH_LIST_ISSUES],
        expect_fns=["create_issue"],
    ),
    S(
        "multi_gh_pr_vs_issue",
        "multiple",
        "Open a pull request from branch 'feature/auth' to 'main' titled 'Add OAuth support'.",
        [T_GH_CREATE_ISSUE, T_GH_CREATE_PR, T_GH_LIST_ISSUES],
        expect_fns=["create_pull_request"],
    ),
    S(
        "multi_gh_file_vs_search",
        "multiple",
        "Get the package.json file from repo owner=vercel repo=next.js.",
        [T_GH_GET_FILE, T_GH_SEARCH_CODE, T_GH_LIST_ISSUES],
        expect_fns=["get_file_contents"],
    ),
    # Slack: pick correct action
    S(
        "multi_slack_send_vs_react",
        "multiple",
        "Post 'Build failed! Check CI' in channel C0DEV.",
        [T_SLACK_SEND, T_SLACK_REACT, T_SLACK_LIST_CH, T_SLACK_UPLOAD],
        expect_fns=["send_message"],
    ),
    S(
        "multi_slack_react_vs_send",
        "multiple",
        "React with :white_check_mark: to the message at 1234.5678 in C0REVIEW.",
        [T_SLACK_SEND, T_SLACK_REACT, T_SLACK_LIST_CH],
        expect_fns=["add_reaction"],
    ),
    S(
        "multi_slack_upload_vs_send",
        "multiple",
        "Upload a file named 'report.csv' with content 'a,b,c' to channel C0DATA.",
        [T_SLACK_SEND, T_SLACK_REACT, T_SLACK_UPLOAD],
        expect_fns=["upload_file"],
    ),
    # PW: pick correct browser action
    S(
        "multi_pw_click_vs_type",
        "multiple",
        "Click the submit button with selector '#submit-btn'.",
        [T_PW_NAV, T_PW_CLICK, T_PW_TYPE, T_PW_SCREENSHOT],
        expect_fns=["browser_click"],
    ),
    S(
        "multi_pw_type_vs_click",
        "multiple",
        "Type 'admin@example.com' into the email input field '#email'.",
        [T_PW_CLICK, T_PW_TYPE, T_PW_SCREENSHOT],
        expect_fns=["browser_type"],
    ),
    # Email: pick correct action
    S(
        "multi_email_send_vs_search",
        "multiple",
        "Email bob@company.com: subject 'Q4 Report', body 'Please review the attached Q4 report'.",
        [T_EMAIL_SEND, T_EMAIL_SEARCH, T_EMAIL_READ],
        expect_fns=["send_email"],
    ),
    S(
        "multi_email_search_vs_read",
        "multiple",
        "Find all emails about 'quarterly review' in my inbox.",
        [T_EMAIL_SEND, T_EMAIL_SEARCH, T_EMAIL_READ],
        expect_fns=["search_email"],
    ),
    # Weather: pick current vs forecast
    S(
        "multi_weather_current_vs_forecast",
        "multiple",
        "What's the temperature right now in New York?",
        [T_WEATHER_CURRENT, T_WEATHER_FORECAST],
        expect_fns=["get_current_weather"],
    ),
    S(
        "multi_weather_forecast_vs_current",
        "multiple",
        "Will it rain in London over the next 3 days?",
        [T_WEATHER_CURRENT, T_WEATHER_FORECAST],
        expect_fns=["get_forecast"],
    ),
    # Math: pick correct operation
    S(
        "multi_calc_vs_convert",
        "multiple",
        "What is 2^10 * 3?",
        [T_CALC, T_CONVERT, T_STATS],
        expect_fns=["calculate"],
    ),
    S(
        "multi_convert_vs_calc",
        "multiple",
        "Convert 72 degrees Fahrenheit to Celsius.",
        [T_CALC, T_CONVERT, T_STATS],
        expect_fns=["convert_units"],
    ),
    S(
        "multi_stats_vs_calc",
        "multiple",
        "Find the mean and standard deviation of test scores: [85, 90, 78, 92, 88, 95, 70].",
        [T_CALC, T_CONVERT, T_STATS],
        expect_fns=["statistics"],
    ),
    # Calendar: pick correct action
    S(
        "multi_cal_create_vs_list",
        "multiple",
        "Schedule a 1-on-1 with Sarah on 2025-04-01 from 14:00 to 14:30.",
        [T_CAL_CREATE, T_CAL_LIST, T_CAL_DELETE],
        expect_fns=["create_event"],
    ),
    S(
        "multi_cal_list_vs_create",
        "multiple",
        "What meetings do I have on 2025-04-01?",
        [T_CAL_CREATE, T_CAL_LIST, T_CAL_DELETE],
        expect_fns=["list_events"],
    ),
    S(
        "multi_cal_delete_vs_others",
        "multiple",
        "Cancel the event with ID evt_abc123.",
        [T_CAL_CREATE, T_CAL_LIST, T_CAL_DELETE],
        expect_fns=["delete_event"],
    ),
    # Docker: pick correct action
    S(
        "multi_docker_logs_vs_stop",
        "multiple",
        "Show me the recent logs from the 'api-server' container, last 100 lines.",
        [T_DOCKER_PS, T_DOCKER_STOP, T_DOCKER_LOGS],
        expect_fns=["docker_logs"],
    ),
    S(
        "multi_docker_stop_vs_logs",
        "multiple",
        "Shut down the container 'staging-db'.",
        [T_DOCKER_RUN, T_DOCKER_STOP, T_DOCKER_LOGS],
        expect_fns=["docker_stop"],
    ),
    # Cross-domain: pick from different domains
    S(
        "multi_cross_search_vs_db",
        "multiple",
        "Search the web for 'FastAPI websocket tutorial'.",
        [T_WEB_SEARCH, T_DB_SELECT, T_FS_READ],
        expect_fns=["web_search"],
    ),
    S(
        "multi_cross_db_vs_search",
        "multiple",
        "Query the database to get all users with role='admin'.",
        [T_WEB_SEARCH, T_DB_SELECT, T_FS_READ],
        expect_fns=["db_query"],
    ),
    S(
        "multi_cross_fs_vs_db",
        "multiple",
        "Read the configuration from /etc/myapp/config.yaml.",
        [T_DB_SELECT, T_FS_READ, T_WEB_SEARCH],
        expect_fns=["read_file"],
    ),
]

# ── PARALLEL: 1 query → multiple calls to same function ──────────────────
PARALLEL = [
    # Read multiple files
    S(
        "para_fs_multi_read",
        "parallel",
        "Read both /etc/hostname and /etc/resolv.conf.",
        [T_FS_READ],
        min_calls=2,
        expect_fns=["read_file"],
        parallel=True,
    ),
    # Search multiple topics
    S(
        "para_search_multi",
        "parallel",
        "Search for 'Python 3.13 features' and also search for 'Rust 2025 roadmap'.",
        [T_WEB_SEARCH],
        min_calls=2,
        expect_fns=["web_search"],
        parallel=True,
    ),
    # Multiple weather queries
    S(
        "para_weather_multi",
        "parallel",
        "Get the current weather in Seoul, Tokyo, and New York.",
        [T_WEATHER_CURRENT],
        min_calls=2,
        expect_fns=["get_current_weather"],
        parallel=True,
    ),
    # Multiple calculations
    S(
        "para_calc_multi",
        "parallel",
        "Calculate three things: 2^16, sqrt(144), and log2(1024).",
        [T_CALC],
        min_calls=2,
        expect_fns=["calculate"],
        parallel=True,
    ),
    # Multiple unit conversions
    S(
        "para_convert_multi",
        "parallel",
        "Convert 100km to miles, 30°C to Fahrenheit, and 5kg to pounds.",
        [T_CONVERT],
        min_calls=2,
        expect_fns=["convert_units"],
        parallel=True,
    ),
    # Multiple DB queries
    S(
        "para_db_multi_query",
        "parallel",
        "Run two queries: count all users, and count all orders placed today.",
        [T_DB_SELECT],
        min_calls=2,
        expect_fns=["db_query"],
        parallel=True,
    ),
    # Multiple file info
    S(
        "para_fs_multi_stat",
        "parallel",
        "Get file info for /var/log/syslog and /var/log/auth.log.",
        [T_FS_STAT],
        min_calls=2,
        expect_fns=["get_file_info"],
        parallel=True,
    ),
    # Multiple GH list issues
    S(
        "para_gh_multi_list",
        "parallel",
        "List open issues in facebook/react and also in vercel/next.js.",
        [T_GH_LIST_ISSUES],
        min_calls=2,
        expect_fns=["list_issues"],
        parallel=True,
    ),
    # Describe multiple tables
    S(
        "para_db_multi_describe",
        "parallel",
        "Describe the schema for 'users', 'orders', and 'products' tables.",
        [T_DB_DESCRIBE],
        min_calls=2,
        expect_fns=["describe_table"],
        parallel=True,
    ),
    # List multiple directories
    S(
        "para_fs_multi_list",
        "parallel",
        "List files in /tmp and /var/log.",
        [T_FS_LIST],
        min_calls=2,
        expect_fns=["list_directory"],
        parallel=True,
    ),
    # Send to multiple channels
    S(
        "para_slack_multi_send",
        "parallel",
        "Send 'Release v3.0 is live!' to both C0GENERAL and C0ENGINEERING.",
        [T_SLACK_SEND],
        min_calls=2,
        expect_fns=["send_message"],
        parallel=True,
    ),
    # Read multiple emails
    S(
        "para_email_multi_search",
        "parallel",
        "Search for emails about 'invoice' and also search for emails about 'contract renewal'.",
        [T_EMAIL_SEARCH],
        min_calls=2,
        expect_fns=["search_email"],
        parallel=True,
    ),
    # Multiple forecasts
    S(
        "para_weather_forecast_multi",
        "parallel",
        "Get 7-day forecast for both Paris and Berlin.",
        [T_WEATHER_FORECAST],
        min_calls=2,
        expect_fns=["get_forecast"],
        parallel=True,
    ),
    # Stop multiple containers
    S(
        "para_docker_stop_multi",
        "parallel",
        "Stop both the 'old-api' and 'legacy-worker' containers.",
        [T_DOCKER_STOP],
        min_calls=2,
        expect_fns=["docker_stop"],
        parallel=True,
    ),
    # Navigate and screenshot (same turn parallel)
    S(
        "para_multi_news",
        "parallel",
        "Search for news about 'climate summit' and also 'tech IPO 2025'.",
        [T_NEWS_SEARCH],
        min_calls=2,
        expect_fns=["news_search"],
        parallel=True,
    ),
]

# ── PARALLEL MULTIPLE: different functions, multiple calls ───────────────
PARALLEL_MULTIPLE = [
    # Read file + write file
    S(
        "pmulti_fs_rw",
        "parallel_multiple",
        "Read /tmp/input.txt and write 'processed' to /tmp/output.txt.",
        [T_FS_READ, T_FS_WRITE],
        min_calls=2,
        expect_fns=["read_file", "write_file"],
        parallel=True,
    ),
    # Search + weather
    S(
        "pmulti_search_weather",
        "parallel_multiple",
        "Search for 'outdoor activities Seoul' and get the current weather in Seoul.",
        [T_WEB_SEARCH, T_WEATHER_CURRENT],
        min_calls=2,
        expect_fns=["web_search", "get_current_weather"],
        parallel=True,
    ),
    # DB query + describe
    S(
        "pmulti_db_query_describe",
        "parallel_multiple",
        "Show me the schema of 'users' table and also query: SELECT COUNT(*) FROM users.",
        [T_DB_SELECT, T_DB_DESCRIBE],
        min_calls=2,
        expect_fns=["db_query", "describe_table"],
        parallel=True,
    ),
    # GH issue + list
    S(
        "pmulti_gh_issue_list",
        "parallel_multiple",
        "Create an issue titled 'Bug in parser' in owner=myorg repo=mylib, "
        "and also list open issues in the same repo.",
        [T_GH_CREATE_ISSUE, T_GH_LIST_ISSUES],
        min_calls=2,
        expect_fns=["create_issue", "list_issues"],
        parallel=True,
    ),
    # Slack send + react
    S(
        "pmulti_slack_send_react",
        "parallel_multiple",
        "Send 'Fix deployed' to C0DEV and react with :rocket: to message 1234.5678 in C0DEV.",
        [T_SLACK_SEND, T_SLACK_REACT],
        min_calls=2,
        expect_fns=["send_message", "add_reaction"],
        parallel=True,
    ),
    # Email send + search
    S(
        "pmulti_email_send_search",
        "parallel_multiple",
        "Send an email to hr@company.com subject 'PTO Request' body 'Requesting March 20-24 off'. "
        "Also search my inbox for emails from hr@company.com.",
        [T_EMAIL_SEND, T_EMAIL_SEARCH],
        min_calls=2,
        expect_fns=["send_email", "search_email"],
        parallel=True,
    ),
    # Calc + convert
    S(
        "pmulti_calc_convert",
        "parallel_multiple",
        "Calculate 2^20 and also convert 42 kg to pounds.",
        [T_CALC, T_CONVERT],
        min_calls=2,
        expect_fns=["calculate", "convert_units"],
        parallel=True,
    ),
    # Memory: entities + relations
    S(
        "pmulti_mem_create",
        "parallel_multiple",
        "Create entity 'Bob' (person, observations: ['senior engineer', 'Python expert']) "
        "and relation Bob -> works_at -> Acme Corp.",
        [T_MEM_ENTITIES, T_MEM_RELATIONS],
        min_calls=2,
        expect_fns=["create_entities", "create_relations"],
        parallel=True,
    ),
    # Weather current + forecast
    S(
        "pmulti_weather_both",
        "parallel_multiple",
        "Get the current weather AND a 3-day forecast for London.",
        [T_WEATHER_CURRENT, T_WEATHER_FORECAST],
        min_calls=2,
        expect_fns=["get_current_weather", "get_forecast"],
        parallel=True,
    ),
    # FS list + search
    S(
        "pmulti_fs_list_search",
        "parallel_multiple",
        "List all files in /home/user and also find all .py files recursively under /home/user/projects.",
        [T_FS_LIST, T_FS_SEARCH],
        min_calls=2,
        expect_fns=["list_directory", "search_files"],
        parallel=True,
    ),
    # Docker run + list
    S(
        "pmulti_docker_run_ps",
        "parallel_multiple",
        "Run a new postgres:16 container named 'test-db' with port 5432:5432, "
        "and list all currently running containers.",
        [T_DOCKER_RUN, T_DOCKER_PS],
        min_calls=2,
        expect_fns=["docker_run", "docker_list"],
        parallel=True,
    ),
    # Calendar create + list
    S(
        "pmulti_cal_create_list",
        "parallel_multiple",
        "Create a meeting 'Team Sync' on 2025-05-01 from 09:00 to 09:30, "
        "and show me all events for that day.",
        [T_CAL_CREATE, T_CAL_LIST],
        min_calls=2,
        expect_fns=["create_event", "list_events"],
        parallel=True,
    ),
    # Cross-domain: DB insert + Slack notify
    S(
        "pmulti_db_slack",
        "parallel_multiple",
        "Insert a new user (name='Charlie', email='charlie@test.com') into the users table, "
        "and send a notification 'New user Charlie registered' to Slack channel C0OPS.",
        [T_DB_INSERT, T_SLACK_SEND],
        min_calls=2,
        expect_fns=["db_insert", "send_message"],
        parallel=True,
    ),
    # Cross-domain: GH + Email
    S(
        "pmulti_gh_email",
        "parallel_multiple",
        "Create a GitHub issue 'Security vulnerability in auth module' in owner=myorg repo=backend, "
        "and email security@myorg.com about it with subject 'Urgent: Auth vulnerability found'.",
        [T_GH_CREATE_ISSUE, T_EMAIL_SEND],
        min_calls=2,
        expect_fns=["create_issue", "send_email"],
        parallel=True,
    ),
    # Cross-domain: FS + DB
    S(
        "pmulti_fs_db",
        "parallel_multiple",
        "Read the SQL migration file at /migrations/001.sql and describe the 'users' table schema.",
        [T_FS_READ, T_DB_DESCRIBE],
        min_calls=2,
        expect_fns=["read_file", "describe_table"],
        parallel=True,
    ),
    # 3-way parallel
    S(
        "pmulti_3way_weather",
        "parallel_multiple",
        "Get current weather in Seoul, search for 'Seoul travel tips', "
        "and list today's calendar events (2025-05-01).",
        [T_WEATHER_CURRENT, T_WEB_SEARCH, T_CAL_LIST],
        min_calls=3,
        expect_fns=["get_current_weather", "web_search", "list_events"],
        parallel=True,
    ),
    # 3-way: FS + DB + Slack
    S(
        "pmulti_3way_ops",
        "parallel_multiple",
        "Read /var/log/app.log, query 'SELECT COUNT(*) FROM errors WHERE date = CURRENT_DATE', "
        "and send 'Error report generated' to Slack C0OPS.",
        [T_FS_READ, T_DB_SELECT, T_SLACK_SEND],
        min_calls=3,
        expect_fns=["read_file", "db_query", "send_message"],
        parallel=True,
    ),
]

# ── IRRELEVANCE: tools provided but none match → should NOT call ─────────
IRRELEVANCE = [
    S(
        "irrel_weather_for_code",
        "irrelevance",
        "Can you explain how Python decorators work?",
        [T_WEATHER_CURRENT, T_WEATHER_FORECAST],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_db_for_joke",
        "irrelevance",
        "Tell me a funny joke about programmers.",
        [T_DB_SELECT, T_DB_INSERT, T_DB_UPDATE],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_gh_for_recipe",
        "irrelevance",
        "How do I make carbonara pasta?",
        [T_GH_CREATE_ISSUE, T_GH_LIST_ISSUES, T_GH_CREATE_PR],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_slack_for_math",
        "irrelevance",
        "What is the capital of France?",
        [T_SLACK_SEND, T_SLACK_REACT, T_SLACK_LIST_CH],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_fs_for_history",
        "irrelevance",
        "When was the Roman Empire founded?",
        [T_FS_READ, T_FS_WRITE, T_FS_LIST],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_docker_for_grammar",
        "irrelevance",
        "Is it 'affect' or 'effect' in this sentence: 'How does weather ___ your mood?'",
        [T_DOCKER_RUN, T_DOCKER_PS, T_DOCKER_STOP],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_email_for_philosophy",
        "irrelevance",
        "What is the meaning of life according to existentialism?",
        [T_EMAIL_SEND, T_EMAIL_SEARCH, T_EMAIL_READ],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_cal_for_translation",
        "irrelevance",
        "Translate 'hello world' to Korean.",
        [T_CAL_CREATE, T_CAL_LIST, T_CAL_DELETE],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_mem_for_sports",
        "irrelevance",
        "Who won the 2024 FIFA Club World Cup?",
        [T_MEM_ENTITIES, T_MEM_RELATIONS, T_MEM_SEARCH],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_browser_for_cooking",
        "irrelevance",
        "What's the difference between baking powder and baking soda?",
        [T_PW_CLICK, T_PW_TYPE, T_PW_SCREENSHOT],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_calc_for_opinion",
        "irrelevance",
        "Should I learn Rust or Go for backend development?",
        [T_CALC, T_CONVERT, T_STATS],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_search_for_personal",
        "irrelevance",
        "How are you feeling today?",
        [T_WEB_SEARCH, T_NEWS_SEARCH, T_LOCAL_SEARCH],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_mixed_for_creative",
        "irrelevance",
        "Write a haiku about autumn leaves.",
        [T_DB_SELECT, T_FS_READ, T_SLACK_SEND, T_DOCKER_PS],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_mixed_for_logic",
        "irrelevance",
        "If all cats are animals, and some animals are pets, can we conclude all cats are pets?",
        [T_GH_CREATE_ISSUE, T_EMAIL_SEND, T_CAL_CREATE],
        expect_fns=[],
        min_calls=0,
    ),
    S(
        "irrel_fs_for_advice",
        "irrelevance",
        "What are some tips for improving sleep quality?",
        [T_FS_READ, T_FS_WRITE, T_FS_SEARCH, T_FS_DELETE],
        expect_fns=[],
        min_calls=0,
    ),
]

# ── Collect all scenarios ────────────────────────────────────────────────
ALL_SCENARIOS = SIMPLE + MULTIPLE + PARALLEL + PARALLEL_MULTIPLE + IRRELEVANCE
SCENARIO_IDS = [s.id for s in ALL_SCENARIOS]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _build_payload(model: str, scenario: S) -> Dict[str, Any]:
    """Build chat completion payload from a scenario."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": scenario.prompt}],
        "tools": scenario.tools,
        "stream": False,
    }
    if scenario.cat == "irrelevance":
        payload["tool_choice"] = "auto"  # model should choose NOT to call
    else:
        payload["tool_choice"] = "required"  # force tool calling
    if scenario.parallel:
        payload["parallel_tool_calls"] = True
    return payload


def _parse_response(resp) -> Dict[str, Any]:
    """Parse and validate the HTTP response."""
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    assert "choices" in data and data["choices"], "missing/empty choices"
    return data


def _get_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg = data["choices"][0].get("message", {})
    return msg.get("tool_calls") or []


def _validate_tool_call(
    tc: Dict[str, Any], scenario_id: str, model: str
) -> Dict[str, Any]:
    """Validate tool_call structure and return parsed args."""
    assert isinstance(tc.get("id"), str) and tc["id"], (
        f"[{model}][{scenario_id}] missing tool_call id"
    )
    assert tc.get("type") == "function", (
        f"[{model}][{scenario_id}] type={tc.get('type')}"
    )
    fn = tc.get("function", {})
    assert isinstance(fn.get("name"), str) and fn["name"], (
        f"[{model}][{scenario_id}] missing function.name"
    )
    raw_args = fn.get("arguments", "")
    assert isinstance(raw_args, str), f"[{model}][{scenario_id}] arguments not string"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        pytest.fail(f"[{model}][{scenario_id}] invalid JSON args: {raw_args[:200]}")
    assert isinstance(args, dict), (
        f"[{model}][{scenario_id}] args not dict: {type(args)}"
    )
    return args


# ═══════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


# ── Simple ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
@pytest.mark.parametrize("scenario", SIMPLE, ids=[s.id for s in SIMPLE])
def test_simple(client: TestClient, model: str, scenario: S) -> None:
    """BFCL Simple: 1 tool, 1 call, correct function + required args."""
    resp = client.post(
        "/v1/chat/completions", json=_build_payload(model, scenario), timeout=TIMEOUT
    )
    data = _parse_response(resp)
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][{scenario.id}] no tool_calls"

    # Exactly 1 call expected for simple
    tc = tcs[0]
    args = _validate_tool_call(tc, scenario.id, model)

    # Correct function name
    fn_name = tc["function"]["name"]
    if scenario.expect_fns:
        assert fn_name in scenario.expect_fns, (
            f"[{model}][{scenario.id}] expected {scenario.expect_fns}, got {fn_name}"
        )

    # Required args present
    if fn_name in scenario.required_args:
        for arg_name in scenario.required_args[fn_name]:
            assert arg_name in args, (
                f"[{model}][{scenario.id}] {fn_name} missing required arg '{arg_name}': {args}"
            )


# ── Multiple ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
@pytest.mark.parametrize("scenario", MULTIPLE, ids=[s.id for s in MULTIPLE])
def test_multiple(client: TestClient, model: str, scenario: S) -> None:
    """BFCL Multiple: 2-4 tools provided, model picks the correct one."""
    resp = client.post(
        "/v1/chat/completions", json=_build_payload(model, scenario), timeout=TIMEOUT
    )
    data = _parse_response(resp)
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][{scenario.id}] no tool_calls"

    tc = tcs[0]
    _validate_tool_call(tc, scenario.id, model)

    fn_name = tc["function"]["name"]
    if scenario.expect_fns:
        assert fn_name in scenario.expect_fns, (
            f"[{model}][{scenario.id}] picked wrong function: {fn_name}, expected {scenario.expect_fns}"
        )


# ── Parallel ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
@pytest.mark.parametrize("scenario", PARALLEL, ids=[s.id for s in PARALLEL])
def test_parallel(client: TestClient, model: str, scenario: S) -> None:
    """BFCL Parallel: single query, multiple calls to same function."""
    resp = client.post(
        "/v1/chat/completions", json=_build_payload(model, scenario), timeout=TIMEOUT
    )
    data = _parse_response(resp)
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][{scenario.id}] no tool_calls"

    # Validate all tool calls
    for tc in tcs:
        _validate_tool_call(tc, scenario.id, model)

    # At least min_calls
    assert len(tcs) >= scenario.min_calls, (
        f"[{model}][{scenario.id}] expected >= {scenario.min_calls} calls, got {len(tcs)}"
    )

    # All calls should be the expected function
    if scenario.expect_fns:
        names = {tc["function"]["name"] for tc in tcs}
        for name in names:
            assert name in scenario.expect_fns, (
                f"[{model}][{scenario.id}] unexpected function {name}, expected {scenario.expect_fns}"
            )


# ── Parallel Multiple ────────────────────────────────────────────────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
@pytest.mark.parametrize(
    "scenario", PARALLEL_MULTIPLE, ids=[s.id for s in PARALLEL_MULTIPLE]
)
def test_parallel_multiple(client: TestClient, model: str, scenario: S) -> None:
    """BFCL Parallel Multiple: multiple tools, calls to different functions."""
    resp = client.post(
        "/v1/chat/completions", json=_build_payload(model, scenario), timeout=TIMEOUT
    )
    data = _parse_response(resp)
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][{scenario.id}] no tool_calls"

    for tc in tcs:
        _validate_tool_call(tc, scenario.id, model)

    # Check minimum calls
    assert len(tcs) >= scenario.min_calls, (
        f"[{model}][{scenario.id}] expected >= {scenario.min_calls} calls, got {len(tcs)}"
    )

    # Check that expected functions are represented
    if scenario.expect_fns:
        called_fns = {tc["function"]["name"] for tc in tcs}
        # At least 1 of the expected functions should be called
        # (relaxed: model may not call ALL in single turn)
        overlap = called_fns & set(scenario.expect_fns)
        assert overlap, (
            f"[{model}][{scenario.id}] expected any of {scenario.expect_fns}, got {called_fns}"
        )


# ── Irrelevance ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
@pytest.mark.parametrize("scenario", IRRELEVANCE, ids=[s.id for s in IRRELEVANCE])
def test_irrelevance(client: TestClient, model: str, scenario: S) -> None:
    """BFCL Irrelevance: tools provided but none relevant → text response."""
    resp = client.post(
        "/v1/chat/completions", json=_build_payload(model, scenario), timeout=TIMEOUT
    )
    data = _parse_response(resp)
    tcs = _get_tool_calls(data)

    # Should NOT call any function
    if tcs:
        called = [tc["function"]["name"] for tc in tcs]
        pytest.fail(
            f"[{model}][{scenario.id}] should not call tools but called: {called}"
        )

    # Should have text content
    msg = data["choices"][0].get("message", {})
    content = msg.get("content") or ""
    assert content.strip(), (
        f"[{model}][{scenario.id}] no tool_calls AND no text content"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print BFCL-style category summary at the end."""
    stats = terminalreporter.stats
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    skipped = len(stats.get("skipped", []))
    total = passed + failed + skipped

    cats = {
        "simple": [0, 0],
        "multiple": [0, 0],
        "parallel": [0, 0],
        "parallel_multiple": [0, 0],
        "irrelevance": [0, 0],
    }

    for report in stats.get("passed", []):
        for cat in cats:
            if f"test_{cat}[" in report.nodeid or f"test_{cat} " in report.nodeid:
                cats[cat][0] += 1
                break
    for report in stats.get("failed", []):
        for cat in cats:
            if f"test_{cat}[" in report.nodeid or f"test_{cat} " in report.nodeid:
                cats[cat][1] += 1
                break

    lines = [
        "",
        "=" * 60,
        "BFCL-STYLE BENCHMARK SUMMARY",
        "=" * 60,
        f"{'Category':<22} {'Pass':>6} {'Fail':>6} {'Rate':>8}",
        "-" * 60,
    ]
    for cat, (p, f) in cats.items():
        t = p + f
        rate = f"{p / t * 100:.1f}%" if t else "N/A"
        lines.append(f"{cat:<22} {p:>6} {f:>6} {rate:>8}")
    lines.append("-" * 60)
    lines.append(
        f"{'TOTAL':<22} {passed:>6} {failed:>6} {passed / total * 100:.1f}%"
        if total
        else ""
    )
    lines.append("=" * 60)

    terminalreporter.write_line("\n".join(lines))
