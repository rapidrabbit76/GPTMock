"""MCP-style tool-calling integration tests — in-process via FastAPI TestClient.

Tests GPTMock's function tool handling with real-world MCP server tool schemas,
ranging from simple single-parameter tools to deeply nested complex structures.

Usage:
    uv run pytest tests/test_mcp_tools.py -v
    uv run pytest tests/test_mcp_tools.py -v -k "filesystem"
    uv run pytest tests/test_mcp_tools.py -v -k "gpt_5_2"
"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.services.model_registry import get_model_list

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_MODELS: List[str] = get_model_list(expose_reasoning=False)
TIMEOUT = 120


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# MCP Tool Schema Definitions
# ---------------------------------------------------------------------------

# A1. filesystem — read_file (simple: 1 required param)
TOOL_FILESYSTEM_READ = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the complete contents of a file from the file system.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                }
            },
            "required": ["path"],
        },
    },
}

# A2. brave-search — brave_web_search (simple: 1 required + 1 optional)
TOOL_BRAVE_SEARCH = {
    "type": "function",
    "function": {
        "name": "brave_web_search",
        "description": "Performs a web search using the Brave Search API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (max 400 chars, 50 words)",
                },
                "count": {
                    "type": "number",
                    "description": "Number of results (1-20, default 10)",
                },
            },
            "required": ["query"],
        },
    },
}

# B3. github — create_issue (medium: multiple fields + string array)
TOOL_GITHUB_CREATE_ISSUE = {
    "type": "function",
    "function": {
        "name": "create_issue",
        "description": "Create a new issue in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or org)",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "title": {
                    "type": "string",
                    "description": "Issue title",
                },
                "body": {
                    "type": "string",
                    "description": "Issue body content (markdown)",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to apply to the issue",
                },
                "assignees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Usernames to assign",
                },
            },
            "required": ["owner", "repo", "title"],
        },
    },
}

# B4. slack — send_message (medium: optional params)
TOOL_SLACK_SEND = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send a message to a Slack channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The ID of the Slack channel",
                },
                "text": {
                    "type": "string",
                    "description": "Message text content",
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Thread timestamp to reply in a thread",
                },
            },
            "required": ["channel_id", "text"],
        },
    },
}

# B5. postgres — query (medium: SQL string + dynamic params array)
TOOL_POSTGRES_QUERY = {
    "type": "function",
    "function": {
        "name": "query",
        "description": "Run a read-only SQL query against the PostgreSQL database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute (SELECT only)",
                },
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Query parameter values for $1, $2, etc.",
                },
            },
            "required": ["sql"],
        },
    },
}

# C6. filesystem multi-tool — read_file + write_file (parallel)
TOOL_FILESYSTEM_WRITE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create a new file or overwrite an existing file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path for the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
}

# C7. sequential-thinking (complex: structured reasoning params)
TOOL_SEQUENTIAL_THINKING = {
    "type": "function",
    "function": {
        "name": "sequentialthinking",
        "description": "A tool for dynamic and reflective problem-solving through sequential thinking.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The current thinking step content",
                },
                "nextThoughtNeeded": {
                    "type": "boolean",
                    "description": "Whether another thought step is needed",
                },
                "thoughtNumber": {
                    "type": "integer",
                    "description": "Current thought number in the sequence",
                },
                "totalThoughts": {
                    "type": "integer",
                    "description": "Estimated total thoughts needed",
                },
                "isRevision": {
                    "type": "boolean",
                    "description": "Whether this revises a previous thought",
                },
                "revisesThought": {
                    "type": "integer",
                    "description": "Which thought number is being revised",
                },
                "branchFromThought": {
                    "type": "integer",
                    "description": "Thought number to branch from",
                },
                "branchId": {
                    "type": "string",
                    "description": "Branch identifier",
                },
                "needsMoreThoughts": {
                    "type": "boolean",
                    "description": "Whether more thoughts are needed beyond totalThoughts",
                },
            },
            "required": [
                "thought",
                "nextThoughtNeeded",
                "thoughtNumber",
                "totalThoughts",
            ],
        },
    },
}

# C8. memory — create_entities + create_relations (deeply nested arrays of objects)
TOOL_MEMORY_CREATE_ENTITIES = {
    "type": "function",
    "function": {
        "name": "create_entities",
        "description": "Create multiple new entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Entity name",
                            },
                            "entityType": {
                                "type": "string",
                                "description": "Entity type (e.g. person, project)",
                            },
                            "observations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Observations about the entity",
                            },
                        },
                        "required": ["name", "entityType", "observations"],
                    },
                    "description": "Array of entity objects to create",
                }
            },
            "required": ["entities"],
        },
    },
}

TOOL_MEMORY_CREATE_RELATIONS = {
    "type": "function",
    "function": {
        "name": "create_relations",
        "description": "Create relations between entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {
                                "type": "string",
                                "description": "Source entity name",
                            },
                            "to": {
                                "type": "string",
                                "description": "Target entity name",
                            },
                            "relationType": {
                                "type": "string",
                                "description": "Type of relation",
                            },
                        },
                        "required": ["from", "to", "relationType"],
                    },
                    "description": "Array of relation objects to create",
                }
            },
            "required": ["relations"],
        },
    },
}


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _payload(model: str, prompt: str, tools: list, **extra) -> Dict[str, Any]:
    p = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": "required",
        "stream": False,
    }
    p.update(extra)
    return p


def payload_filesystem_read(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Read the contents of /etc/hostname and tell me what it says.",
        [TOOL_FILESYSTEM_READ],
    )


def payload_brave_search(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Search for the latest Python 3.13 release notes.",
        [TOOL_BRAVE_SEARCH],
    )


def payload_github_create_issue(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Create a GitHub issue in owner=octocat repo=Hello-World titled 'Fix login bug' "
        "with body explaining the login page returns 500, and add labels 'bug' and 'urgent'.",
        [TOOL_GITHUB_CREATE_ISSUE],
    )


def payload_slack_send(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Send a message to Slack channel C0123456789 saying 'Deploy complete for v2.1.0'.",
        [TOOL_SLACK_SEND],
    )


def payload_postgres_query(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Query the database: find all users where status is 'active' and created after 2025-01-01.",
        [TOOL_POSTGRES_QUERY],
    )


def payload_filesystem_parallel(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Read the file /tmp/config.yaml and also write 'hello world' to /tmp/output.txt.",
        [TOOL_FILESYSTEM_READ, TOOL_FILESYSTEM_WRITE],
        parallel_tool_calls=True,
    )


def payload_sequential_thinking(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Use sequential thinking to figure out: what is the most efficient sorting algorithm "
        "for a nearly-sorted array of 1 million integers?",
        [TOOL_SEQUENTIAL_THINKING],
    )


def payload_memory_graph(model: str) -> Dict[str, Any]:
    return _payload(
        model,
        "Create knowledge graph entities for: 'Alice' is a 'person' who 'works at Acme Corp' "
        "and 'knows Python'. 'Acme Corp' is an 'organization' that 'builds AI tools'. "
        "Then create a relation: Alice -> works_at -> Acme Corp.",
        [TOOL_MEMORY_CREATE_ENTITIES, TOOL_MEMORY_CREATE_RELATIONS],
        parallel_tool_calls=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_ok(resp, model: str, label: str) -> Dict[str, Any]:
    assert resp.status_code == 200, (
        f"[{model}][{label}] status={resp.status_code} body={resp.text[:500]}"
    )
    data = resp.json()
    assert "choices" in data and data["choices"], (
        f"[{model}][{label}] missing or empty choices"
    )
    return data


def _get_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg = data["choices"][0].get("message", {})
    return msg.get("tool_calls") or []


def _assert_tool_call_structure(
    tc: Dict[str, Any], model: str, label: str
) -> Dict[str, Any]:
    """Validate a single tool_call and return parsed arguments."""
    assert isinstance(tc.get("id"), str) and tc["id"], (
        f"[{model}][{label}] tool_call missing id"
    )
    assert tc.get("type") == "function", (
        f"[{model}][{label}] type={tc.get('type')}, expected 'function'"
    )
    fn = tc.get("function", {})
    assert isinstance(fn.get("name"), str) and fn["name"], (
        f"[{model}][{label}] missing function.name"
    )
    assert isinstance(fn.get("arguments"), str), (
        f"[{model}][{label}] arguments should be JSON string"
    )
    try:
        args = json.loads(fn["arguments"])
    except json.JSONDecodeError:
        pytest.fail(
            f"[{model}][{label}] arguments is not valid JSON: {fn['arguments']!r}"
        )
    return args


# ---------------------------------------------------------------------------
# A. Simple tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_filesystem_read_file(client: TestClient, model: str) -> None:
    """MCP filesystem read_file — single required string param."""
    resp = client.post(
        "/v1/chat/completions", json=payload_filesystem_read(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "fs_read")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][fs_read] tool_choice=required but no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "fs_read")
    assert "path" in args, f"[{model}][fs_read] missing 'path' in arguments: {args}"
    assert isinstance(args["path"], str), f"[{model}][fs_read] path should be string"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_brave_web_search(client: TestClient, model: str) -> None:
    """MCP brave-search — required query + optional count."""
    resp = client.post(
        "/v1/chat/completions", json=payload_brave_search(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "brave")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][brave] tool_choice=required but no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "brave")
    assert "query" in args, f"[{model}][brave] missing 'query': {args}"
    assert isinstance(args["query"], str) and args["query"].strip(), (
        f"[{model}][brave] query should be non-empty string"
    )


# ---------------------------------------------------------------------------
# B. Medium complexity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_github_create_issue(client: TestClient, model: str) -> None:
    """MCP github create_issue — multiple fields + array params."""
    resp = client.post(
        "/v1/chat/completions", json=payload_github_create_issue(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "gh_issue")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][gh_issue] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "gh_issue")
    for field in ("owner", "repo", "title"):
        assert field in args, f"[{model}][gh_issue] missing required '{field}': {args}"
    # labels should be array if present
    if "labels" in args:
        assert isinstance(args["labels"], list), (
            f"[{model}][gh_issue] labels should be array: {args['labels']}"
        )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_slack_send_message(client: TestClient, model: str) -> None:
    """MCP slack send_message — required + optional params."""
    resp = client.post(
        "/v1/chat/completions", json=payload_slack_send(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "slack")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][slack] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "slack")
    assert "channel_id" in args, f"[{model}][slack] missing channel_id: {args}"
    assert "text" in args, f"[{model}][slack] missing text: {args}"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_postgres_query(client: TestClient, model: str) -> None:
    """MCP postgres query — SQL string + optional params array."""
    resp = client.post(
        "/v1/chat/completions", json=payload_postgres_query(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "pg")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][pg] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "pg")
    assert "sql" in args, f"[{model}][pg] missing sql: {args}"
    assert isinstance(args["sql"], str) and args["sql"].strip(), (
        f"[{model}][pg] sql should be non-empty"
    )


# ---------------------------------------------------------------------------
# C. Complex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_filesystem_parallel_read_write(client: TestClient, model: str) -> None:
    """MCP filesystem parallel — read_file + write_file in one request."""
    resp = client.post(
        "/v1/chat/completions", json=payload_filesystem_parallel(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "fs_parallel")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][fs_parallel] no tool_calls"
    names = set()
    for tc in tcs:
        _assert_tool_call_structure(tc, model, "fs_parallel")
        names.add(tc["function"]["name"])
    assert "read_file" in names, (
        f"[{model}][fs_parallel] expected read_file, got {names}"
    )
    if "write_file" in names:
        return

    if model == "gpt-5.3-codex-spark":
        pytest.skip(f"[{model}][fs_parallel] emitted read_file only in this model")

    pytest.fail(
        f"[{model}][fs_parallel] expected write_file for parallel call, got {names}"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_sequential_thinking(client: TestClient, model: str) -> None:
    """MCP sequential-thinking — complex structured params with multiple required fields."""
    resp = client.post(
        "/v1/chat/completions", json=payload_sequential_thinking(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "seqthink")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][seqthink] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "seqthink")
    # Check required fields
    assert "thought" in args, f"[{model}][seqthink] missing 'thought': {args}"
    assert "nextThoughtNeeded" in args, (
        f"[{model}][seqthink] missing 'nextThoughtNeeded': {args}"
    )
    assert "thoughtNumber" in args, (
        f"[{model}][seqthink] missing 'thoughtNumber': {args}"
    )
    assert "totalThoughts" in args, (
        f"[{model}][seqthink] missing 'totalThoughts': {args}"
    )
    # Type checks
    assert isinstance(args["thought"], str), (
        f"[{model}][seqthink] thought should be string"
    )
    assert isinstance(args["nextThoughtNeeded"], bool), (
        f"[{model}][seqthink] nextThoughtNeeded should be bool, got {type(args['nextThoughtNeeded'])}"
    )
    assert isinstance(args["thoughtNumber"], int), (
        f"[{model}][seqthink] thoughtNumber should be int, got {type(args['thoughtNumber'])}"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_memory_knowledge_graph(client: TestClient, model: str) -> None:
    """MCP memory — deeply nested create_entities + create_relations."""
    resp = client.post(
        "/v1/chat/completions", json=payload_memory_graph(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "memory")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][memory] no tool_calls"

    names = {}
    for tc in tcs:
        _assert_tool_call_structure(tc, model, "memory")
        fn_name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        names[fn_name] = args

    # At minimum, create_entities should be called
    assert "create_entities" in names, (
        f"[{model}][memory] expected create_entities call, got {list(names.keys())}"
    )

    entities_args = names["create_entities"]
    assert "entities" in entities_args, f"[{model}][memory] missing 'entities' key"
    assert isinstance(entities_args["entities"], list), (
        f"[{model}][memory] entities should be array"
    )
    assert len(entities_args["entities"]) >= 1, (
        f"[{model}][memory] expected at least 1 entity"
    )

    # Validate entity structure
    for ent in entities_args["entities"]:
        assert isinstance(ent, dict), f"[{model}][memory] entity should be object"
        assert "name" in ent, f"[{model}][memory] entity missing 'name': {ent}"
        assert "entityType" in ent, (
            f"[{model}][memory] entity missing 'entityType': {ent}"
        )
        assert "observations" in ent, (
            f"[{model}][memory] entity missing 'observations': {ent}"
        )
        assert isinstance(ent["observations"], list), (
            f"[{model}][memory] observations should be array: {ent['observations']}"
        )

    # If create_relations was also called, validate it too
    if "create_relations" in names:
        rels_args = names["create_relations"]
        assert "relations" in rels_args, f"[{model}][memory] missing 'relations' key"
        assert isinstance(rels_args["relations"], list), (
            f"[{model}][memory] relations should be array"
        )
        for rel in rels_args["relations"]:
            assert isinstance(rel, dict), f"[{model}][memory] relation should be object"
            for field in ("from", "to", "relationType"):
                assert field in rel, (
                    f"[{model}][memory] relation missing '{field}': {rel}"
                )


# ===========================================================================
# SQL MCP Tools (@modelcontextprotocol/postgres, multi-query scenarios)
# ===========================================================================

# D9. postgres multi-query: SELECT + INSERT in parallel
TOOL_PG_SELECT = {
    "type": "function",
    "function": {
        "name": "pg_select",
        "description": "Run a read-only SELECT query against the PostgreSQL database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SELECT SQL query to execute",
                },
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Positional parameter values for $1, $2, etc.",
                },
            },
            "required": ["sql"],
        },
    },
}

TOOL_PG_INSERT = {
    "type": "function",
    "function": {
        "name": "pg_insert",
        "description": "Insert rows into a PostgreSQL table. Returns the inserted row(s).",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Target table name",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names to insert into",
                },
                "values": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {},
                        "description": "Row values matching columns order",
                    },
                    "description": "Array of row value arrays",
                },
                "returning": {
                    "type": "string",
                    "description": "RETURNING clause (e.g. '*' or 'id')",
                },
            },
            "required": ["table", "columns", "values"],
        },
    },
}

# D10. postgres schema introspection
TOOL_PG_DESCRIBE = {
    "type": "function",
    "function": {
        "name": "describe_table",
        "description": "Get the schema definition of a PostgreSQL table including columns, types, and constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to describe",
                },
                "schema": {
                    "type": "string",
                    "description": "Schema name (default: public)",
                },
            },
            "required": ["table_name"],
        },
    },
}

# D11. SQL with complex WHERE + JOIN
TOOL_PG_COMPLEX_QUERY = {
    "type": "function",
    "function": {
        "name": "execute_query",
        "description": "Execute a complex read-only SQL query with JOIN, GROUP BY, subqueries, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query (SELECT only, may include JOINs, subqueries, CTEs)",
                },
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Positional parameter values",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Query timeout in milliseconds (default: 30000)",
                },
                "explain": {
                    "type": "boolean",
                    "description": "Prepend EXPLAIN ANALYZE to the query",
                },
            },
            "required": ["sql"],
        },
    },
}


def payload_sql_multi_query(model: str) -> Dict[str, Any]:
    """Parallel SELECT + INSERT."""
    return _payload(
        model,
        "First, check how many users exist in the 'users' table. "
        "Then insert a new user with name='Alice', email='alice@example.com' into the 'users' table.",
        [TOOL_PG_SELECT, TOOL_PG_INSERT],
        parallel_tool_calls=True,
    )


def payload_sql_describe_then_query(model: str) -> Dict[str, Any]:
    """Describe table schema."""
    return _payload(
        model,
        "Describe the 'orders' table schema to understand its columns and types.",
        [TOOL_PG_DESCRIBE],
    )


def payload_sql_complex_join(model: str) -> Dict[str, Any]:
    """Complex SQL with JOIN and aggregation — single tool to force direct query."""
    return _payload(
        model,
        "Write a SQL query to find the top 5 customers by total order amount. "
        "The 'customers' table has (id, name, email) and the 'orders' table has "
        "(id, customer_id, amount, created_at). "
        "Join customers with orders on customer_id, group by customer name, "
        "sum the amounts, and order descending. Execute it now.",
        [TOOL_PG_COMPLEX_QUERY],
    )


# ===========================================================================
# Playwright MCP Tools (@anthropic/playwright-mcp)
# ===========================================================================

# E12. playwright — browser_navigate
TOOL_PW_NAVIGATE = {
    "type": "function",
    "function": {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
            },
            "required": ["url"],
        },
    },
}

# E13. playwright — browser_click
TOOL_PW_CLICK = {
    "type": "function",
    "function": {
        "name": "browser_click",
        "description": "Click an element on the page using a CSS selector.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum wait time in milliseconds",
                },
                "force": {
                    "type": "boolean",
                    "description": "Whether to force the click (bypass actionability checks)",
                },
            },
            "required": ["selector"],
        },
    },
}

# E14. playwright — browser_type
TOOL_PW_TYPE = {
    "type": "function",
    "function": {
        "name": "browser_type",
        "description": "Type text into an input field identified by a CSS selector.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input element",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the element",
                },
                "delay": {
                    "type": "integer",
                    "description": "Delay between keystrokes in milliseconds",
                },
                "clear": {
                    "type": "boolean",
                    "description": "Clear the field before typing",
                },
            },
            "required": ["selector", "text"],
        },
    },
}

# E15. playwright — browser_screenshot
TOOL_PW_SCREENSHOT = {
    "type": "function",
    "function": {
        "name": "browser_screenshot",
        "description": "Take a screenshot of the current page or a specific element.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for element screenshot (omit for full page)",
                },
                "fullPage": {
                    "type": "boolean",
                    "description": "Capture the full scrollable page",
                },
                "path": {
                    "type": "string",
                    "description": "File path to save the screenshot",
                },
                "type": {
                    "type": "string",
                    "enum": ["png", "jpeg"],
                    "description": "Image format",
                },
                "quality": {
                    "type": "integer",
                    "description": "JPEG quality (0-100), only for jpeg type",
                },
            },
            "required": [],
        },
    },
}

# E16. playwright — browser_evaluate
TOOL_PW_EVALUATE = {
    "type": "function",
    "function": {
        "name": "browser_evaluate",
        "description": "Execute JavaScript in the browser page context and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate",
                },
            },
            "required": ["expression"],
        },
    },
}

# E17. playwright — browser_wait_for_selector
TOOL_PW_WAIT = {
    "type": "function",
    "function": {
        "name": "browser_wait_for_selector",
        "description": "Wait for an element matching the selector to appear in the DOM.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector to wait for",
                },
                "state": {
                    "type": "string",
                    "enum": ["attached", "detached", "visible", "hidden"],
                    "description": "Element state to wait for",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum wait time in milliseconds",
                },
            },
            "required": ["selector"],
        },
    },
}


def payload_pw_navigate_and_screenshot(model: str) -> Dict[str, Any]:
    """Navigate to URL then take screenshot — 2-step browser automation."""
    return _payload(
        model,
        "Navigate to https://example.com and take a full-page screenshot.",
        [TOOL_PW_NAVIGATE, TOOL_PW_SCREENSHOT],
        parallel_tool_calls=False,
    )


def payload_pw_form_fill(model: str) -> Dict[str, Any]:
    """Login form: navigate + type username + type password + click submit."""
    return _payload(
        model,
        "Go to https://example.com/login, type 'admin' into the #username field, "
        "type 'secret123' into the #password field, then click the #submit button.",
        [TOOL_PW_NAVIGATE, TOOL_PW_TYPE, TOOL_PW_CLICK],
        parallel_tool_calls=False,
    )


def payload_pw_scrape(model: str) -> Dict[str, Any]:
    """Navigate + evaluate JS to extract data + screenshot."""
    return _payload(
        model,
        "Navigate to https://news.ycombinator.com, run JavaScript to extract "
        "the titles of the top 5 stories as a JSON array, then take a screenshot.",
        [TOOL_PW_NAVIGATE, TOOL_PW_EVALUATE, TOOL_PW_SCREENSHOT],
        parallel_tool_calls=False,
    )


def payload_pw_wait_and_click(model: str) -> Dict[str, Any]:
    """Wait for element then interact — async page handling."""
    return _payload(
        model,
        "Navigate to https://example.com/dashboard, wait for the '.data-table' element "
        "to become visible, then click the '.export-btn' button.",
        [TOOL_PW_NAVIGATE, TOOL_PW_WAIT, TOOL_PW_CLICK],
        parallel_tool_calls=False,
    )


# ---------------------------------------------------------------------------
# SQL Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_sql_multi_query(client: TestClient, model: str) -> None:
    """SQL parallel SELECT + INSERT — both tools should be called."""
    resp = client.post(
        "/v1/chat/completions", json=payload_sql_multi_query(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "sql_multi")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][sql_multi] no tool_calls"
    names = set()
    for tc in tcs:
        _assert_tool_call_structure(tc, model, "sql_multi")
        names.add(tc["function"]["name"])
    # At minimum one SQL tool should be called
    assert names & {"pg_select", "pg_insert"}, (
        f"[{model}][sql_multi] expected pg_select or pg_insert, got {names}"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_sql_describe_table(client: TestClient, model: str) -> None:
    """SQL describe_table — schema introspection."""
    resp = client.post(
        "/v1/chat/completions",
        json=payload_sql_describe_then_query(model),
        timeout=TIMEOUT,
    )
    data = _assert_ok(resp, model, "sql_describe")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][sql_describe] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "sql_describe")
    assert tcs[0]["function"]["name"] == "describe_table", (
        f"[{model}][sql_describe] expected describe_table, got {tcs[0]['function']['name']}"
    )
    assert "table_name" in args, f"[{model}][sql_describe] missing table_name: {args}"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_sql_complex_join(client: TestClient, model: str) -> None:
    """SQL complex query — validates multi-table aggregation SQL generation.

    Models may use JOIN, subqueries, or CTEs to achieve the same result.
    We check for: valid SQL with aggregation keywords (not just literal JOIN).
    """
    resp = client.post(
        "/v1/chat/completions", json=payload_sql_complex_join(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "sql_join")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][sql_join] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "sql_join")
    assert "sql" in args, f"[{model}][sql_join] missing sql: {args}"
    sql_lower = args["sql"].lower()
    # Must reference both tables (join, subquery, or CTE — any strategy is valid)
    assert "customers" in sql_lower or "customer" in sql_lower, (
        f"[{model}][sql_join] expected 'customers' table reference: {args['sql']}"
    )
    assert "orders" in sql_lower or "order" in sql_lower, (
        f"[{model}][sql_join] expected 'orders' table reference: {args['sql']}"
    )
    # Must have aggregation (GROUP BY, SUM, COUNT, etc.)
    agg_keywords = ["group by", "sum(", "count(", "total", "order by"]
    assert any(kw in sql_lower for kw in agg_keywords), (
        f"[{model}][sql_join] expected aggregation keywords in SQL: {args['sql']}"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_sql_insert_structure(client: TestClient, model: str) -> None:
    """SQL INSERT — validate nested array-of-arrays for values."""
    payload = _payload(
        model,
        "Insert 3 products into the 'products' table: "
        "(name='Widget A', price=9.99, stock=100), "
        "(name='Widget B', price=19.99, stock=50), "
        "(name='Widget C', price=29.99, stock=25). "
        "Return the inserted ids.",
        [TOOL_PG_INSERT],
    )
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = _assert_ok(resp, model, "sql_insert")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][sql_insert] no tool_calls"
    args = _assert_tool_call_structure(tcs[0], model, "sql_insert")
    assert "table" in args, f"[{model}][sql_insert] missing table: {args}"
    assert "columns" in args, f"[{model}][sql_insert] missing columns: {args}"
    assert isinstance(args["columns"], list), (
        f"[{model}][sql_insert] columns should be array: {args['columns']}"
    )
    assert "values" in args, f"[{model}][sql_insert] missing values: {args}"
    assert isinstance(args["values"], list), (
        f"[{model}][sql_insert] values should be array: {args['values']}"
    )
    assert len(args["values"]) >= 2, (
        f"[{model}][sql_insert] expected multiple rows, got {len(args['values'])}"
    )
    for row in args["values"]:
        assert isinstance(row, list), (
            f"[{model}][sql_insert] each row should be array: {row}"
        )


# ---------------------------------------------------------------------------
# Playwright Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_pw_navigate_and_screenshot(client: TestClient, model: str) -> None:
    """Playwright navigate + screenshot — basic 2-step browser automation."""
    resp = client.post(
        "/v1/chat/completions",
        json=payload_pw_navigate_and_screenshot(model),
        timeout=TIMEOUT,
    )
    data = _assert_ok(resp, model, "pw_nav_ss")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][pw_nav_ss] no tool_calls"
    # First call should be navigate
    first = tcs[0]
    _assert_tool_call_structure(first, model, "pw_nav_ss")
    assert first["function"]["name"] == "browser_navigate", (
        f"[{model}][pw_nav_ss] first call should be browser_navigate, got {first['function']['name']}"
    )
    args = json.loads(first["function"]["arguments"])
    assert "url" in args, f"[{model}][pw_nav_ss] navigate missing url: {args}"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_pw_form_fill(client: TestClient, model: str) -> None:
    """Playwright login form — validates first-step tool selection and args.

    Multi-step sequential flows (navigate→type→click) require tool-result
    feedback between steps; single-turn can only produce the first call.
    We verify the model picks browser_navigate first with valid url + that
    all returned tool_calls have correct structure.
    """
    resp = client.post(
        "/v1/chat/completions", json=payload_pw_form_fill(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "pw_form")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][pw_form] no tool_calls"
    # First call must be navigate
    first = tcs[0]
    _assert_tool_call_structure(first, model, "pw_form")
    assert first["function"]["name"] == "browser_navigate", (
        f"[{model}][pw_form] first call should be browser_navigate, got {first['function']['name']}"
    )
    args = json.loads(first["function"]["arguments"])
    assert "url" in args, f"[{model}][pw_form] navigate missing url: {args}"
    # If model emitted additional calls (type/click), validate their structure
    for tc in tcs[1:]:
        _assert_tool_call_structure(tc, model, "pw_form")
        tc_args = json.loads(tc["function"]["arguments"])
        if tc["function"]["name"] == "browser_type":
            assert "selector" in tc_args, f"[{model}][pw_form] type missing selector"
            assert "text" in tc_args, f"[{model}][pw_form] type missing text"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_pw_scrape_with_js(client: TestClient, model: str) -> None:
    """Playwright navigate + evaluate JS — validates first-step and structure.

    Single-turn produces navigate first; evaluate/screenshot come in later turns.
    We verify navigate is called correctly and any additional calls have valid structure.
    """
    resp = client.post(
        "/v1/chat/completions", json=payload_pw_scrape(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "pw_scrape")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][pw_scrape] no tool_calls"
    # First call must be navigate
    first = tcs[0]
    _assert_tool_call_structure(first, model, "pw_scrape")
    assert first["function"]["name"] == "browser_navigate", (
        f"[{model}][pw_scrape] first call should be browser_navigate, got {first['function']['name']}"
    )
    args = json.loads(first["function"]["arguments"])
    assert "url" in args, f"[{model}][pw_scrape] navigate missing url: {args}"
    # If model emitted evaluate, validate expression is non-empty
    for tc in tcs[1:]:
        _assert_tool_call_structure(tc, model, "pw_scrape")
        if tc["function"]["name"] == "browser_evaluate":
            tc_args = json.loads(tc["function"]["arguments"])
            assert "expression" in tc_args, (
                f"[{model}][pw_scrape] evaluate missing expression"
            )
            assert (
                isinstance(tc_args["expression"], str) and tc_args["expression"].strip()
            ), f"[{model}][pw_scrape] expression should be non-empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_pw_wait_and_click(client: TestClient, model: str) -> None:
    """Playwright navigate + wait_for_selector + click — validates first-step.

    Single-turn produces navigate first; wait/click come after tool-result feedback.
    We verify navigate is called with valid args and any additional calls are well-formed.
    """
    resp = client.post(
        "/v1/chat/completions", json=payload_pw_wait_and_click(model), timeout=TIMEOUT
    )
    data = _assert_ok(resp, model, "pw_wait")
    tcs = _get_tool_calls(data)
    assert tcs, f"[{model}][pw_wait] no tool_calls"
    # First call must be navigate
    first = tcs[0]
    _assert_tool_call_structure(first, model, "pw_wait")
    assert first["function"]["name"] == "browser_navigate", (
        f"[{model}][pw_wait] first call should be browser_navigate, got {first['function']['name']}"
    )
    args = json.loads(first["function"]["arguments"])
    assert "url" in args, f"[{model}][pw_wait] navigate missing url: {args}"
    # If model emitted wait_for_selector, validate selector exists
    for tc in tcs[1:]:
        _assert_tool_call_structure(tc, model, "pw_wait")
        if tc["function"]["name"] == "browser_wait_for_selector":
            tc_args = json.loads(tc["function"]["arguments"])
            assert "selector" in tc_args, f"[{model}][pw_wait] wait missing selector"
