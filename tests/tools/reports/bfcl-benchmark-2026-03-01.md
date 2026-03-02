# BFCL-Style Function-Calling Benchmark Report

**Date**: 2026-03-01  
**File**: `tests/test_bfcl_benchmark.py`  
**Runtime**: 3,937.91s (65 min 37 sec)  
**Total Tests**: 1,098 (122 scenarios × 9 models)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Passed** | 944 |
| **Failed** | 154 |
| **Pass Rate** | **85.97%** |

The benchmark adopts [Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) category structure with GPTMock-registered model aliases. All tests use synchronous `TestClient` (in-process) with `tool_choice: "required"` except Irrelevance tests which use `tool_choice: "auto"`.

---

## Category Results

| Category | Scenarios | Tests | Passed | Failed | Pass Rate |
|----------|-----------|-------|--------|--------|-----------|
| **Simple** | 41 | 369 | 353 | 16 | 95.7% |
| **Multiple** | 34 | 306 | 248 | 58 | 81.0% |
| **Parallel** | 15 | 135 | 118 | 17 | 87.4% |
| **Parallel Multiple** | 17 | 153 | 93 | 60 | 60.8% |
| **Irrelevance** | 15 | 135 | 132 | 3 | 97.8% |

### Category Descriptions

- **Simple**: 1 tool provided, 1 call expected. Validates correct function selection and required argument presence.
- **Multiple**: 2–4 tools provided, model must pick the correct 1. Tests function discrimination.
- **Parallel**: Single query requiring multiple calls to the same function (e.g., "read both files").
- **Parallel Multiple**: Multiple different tools, model should call ≥2 distinct functions in one turn.
- **Irrelevance**: Tools provided but none relevant to the query. Model should respond with text only.

---

## Model Results

| Model | Passed | Failed | Pass Rate | Rank |
|-------|--------|--------|-----------|------|
| gpt-5.2-codex | 111 | 11 | **91.0%** | 🥇 1 |
| gpt-5.3-codex | 107 | 15 | 87.7% | 🥈 2 |
| gpt-5.1-codex-max | 107 | 15 | 87.7% | 🥈 2 |
| gpt-5 | 106 | 16 | 86.9% | 4 |
| gpt-5.1 | 106 | 16 | 86.9% | 4 |
| gpt-5.2 | 104 | 18 | 85.2% | 6 |
| gpt-5.3-codex-spark | 102 | 20 | 83.6% | 7 |
| gpt-5.1-codex | 101 | 21 | 82.8% | 8 |
| gpt-5-codex | 100 | 22 | **82.0%** | 9 |

---

## Model × Category Heatmap

| Model | Simple (41) | Multiple (34) | Parallel (15) | Par.Multi (17) | Irrelevance (15) |
|-------|-------------|---------------|---------------|----------------|-------------------|
| gpt-5 | 39 ✓ / 2 ✗ | 29 ✓ / 5 ✗ | 15 ✓ / 0 ✗ | 11 ✓ / 6 ✗ | 15 ✓ / 0 ✗ |
| gpt-5.1 | 40 ✓ / 1 ✗ | 28 ✓ / 6 ✗ | 14 ✓ / 1 ✗ | 12 ✓ / 5 ✗ | 15 ✓ / 0 ✗ |
| gpt-5.2 | 39 ✓ / 2 ✗ | 27 ✓ / 7 ✗ | 13 ✓ / 2 ✗ | 12 ✓ / 5 ✗ | 15 ✓ / 0 ✗ |
| gpt-5-codex | 39 ✓ / 2 ✗ | 27 ✓ / 7 ✗ | 10 ✓ / 5 ✗ | 11 ✓ / 6 ✗ | 14 ✓ / 1 ✗ |
| gpt-5.2-codex | 39 ✓ / 2 ✗ | 30 ✓ / 4 ✗ | 15 ✓ / 0 ✗ | 14 ✓ / 3 ✗ | 15 ✓ / 0 ✗ |
| gpt-5.3-codex | 39 ✓ / 2 ✗ | 29 ✓ / 5 ✗ | 14 ✓ / 1 ✗ | 12 ✓ / 5 ✗ | 15 ✓ / 0 ✗ |
| gpt-5.3-codex-spark | 38 ✓ / 3 ✗ | 26 ✓ / 8 ✗ | 14 ✓ / 1 ✗ | 10 ✓ / 7 ✗ | 15 ✓ / 0 ✗ |
| gpt-5.1-codex | 39 ✓ / 2 ✗ | 27 ✓ / 7 ✗ | 13 ✓ / 2 ✗ | 10 ✓ / 7 ✗ | 13 ✓ / 2 ✗ |
| gpt-5.1-codex-max | 39 ✓ / 2 ✗ | 28 ✓ / 6 ✗ | 12 ✓ / 3 ✗ | 10 ✓ / 7 ✗ | 15 ✓ / 0 ✗ |

> Note: Per-model category counts are approximated from failure list distribution.

---

## Tool Domains Covered (13)

| Domain | Tools | Count |
|--------|-------|-------|
| Filesystem | read_file, write_file, list_directory, move_file, search_files, delete_file, create_directory, get_file_info | 8 |
| Database | db_query, db_insert, db_update, db_delete, describe_table, list_tables | 6 |
| GitHub | create_issue, list_issues, create_pull_request, get_file_contents, search_code | 5 |
| Browser | browser_navigate, browser_click, browser_type, browser_screenshot, browser_evaluate | 5 |
| Slack | send_message, list_channels, add_reaction, upload_file | 4 |
| Docker | docker_run, docker_list, docker_stop, docker_logs | 4 |
| Search | web_search, local_search, news_search | 3 |
| Memory | create_entities, create_relations, search_nodes | 3 |
| Email | send_email, search_email, read_email | 3 |
| Math | calculate, convert_units, statistics | 3 |
| Calendar | create_event, list_events, delete_event | 3 |
| Weather | get_current_weather, get_forecast | 2 |

---

## Failure Analysis

### All-Model Failures (9/9 models fail)

| Scenario | Category | Root Cause |
|----------|----------|------------|
| `simple_db_update` | Simple | Models pack SET clause into the `where` argument instead of using the separate `set` parameter. Schema has `set: object` + `where: string` but models merge them. |
| `pmulti_fs_rw` | Par.Multi | "Read X and write Y" — all models call only `read_file`, deferring `write_file` until read result is available. Sequential dependency. |
| `pmulti_fs_db` | Par.Multi | "Read migration file and describe table" — similar to above, models read first, plan to describe after. |
| `multi_gh_issue_vs_pr` | Multiple | "Report a bug" — all models call `list_issues` (to check for duplicates) instead of `create_issue`. Reasonable model behavior. |

### High-Failure Scenarios (7–8/9 models fail)

| Scenario | Category | Fails | Root Cause |
|----------|----------|-------|------------|
| `pmulti_docker_run_ps` | Par.Multi | 8/9 | Models prefer sequential: run first, then list. |
| `multi_db_update_vs_delete` | Multiple | 8/9 | "Change status" — models pick `db_query` (raw UPDATE SQL) instead of structured `db_update`. |
| `multi_convert_vs_calc` | Multiple | 8/9 | "Convert 72°F to Celsius" — models use `calculate` (math expression) instead of `convert_units`. Both arguably correct. |
| `pmulti_cal_create_list` | Par.Multi | 7/9 | Create + list events: models create first, expect to list after. |
| `para_calc_multi` | Parallel | 7/9 | "Calculate 3 things" — models produce 1 call with all expressions combined. |
| `multi_gh_pr_vs_issue` | Multiple | 7/9 | "Open a PR" — models call `create_issue` or `list_issues` for context first. |

### Sporadic Failures (1–4/9 models fail)

| Scenario | Category | Fails | Affected Models |
|----------|----------|-------|-----------------|
| `simple_calc` | Simple | 5/9 | gpt-5, 5.1, 5.2, 5.3-codex, 5.3-codex-spark |
| `multi_db_insert_vs_others` | Multiple | 5/9 | gpt-5.1, 5.2, 5-codex, 5.3-spark, 5.1-max |
| `pmulti_mem_create` | Par.Multi | 5/9 | gpt-5, 5.3-codex, 5.3-spark, 5.1-codex, 5.1-max |
| `pmulti_gh_issue_list` | Par.Multi | 5/9 | gpt-5, 5-codex, 5.3-codex, 5.3-spark, 5.1-codex |
| `multi_calc_vs_convert` | Multiple | 5/9 | gpt-5, 5.1, 5.2, 5.3-codex, 5.3-spark |
| `multi_fs_search_vs_list` | Multiple | 4/9 | gpt-5.2, 5-codex, 5.1-codex, 5.1-max |
| `pmulti_db_slack` | Par.Multi | 4/9 | gpt-5.1, 5.3-codex, 5.3-spark, 5.1-codex |
| `multi_email_send_vs_search` | Multiple | 3/9 | gpt-5, 5.2, 5.2-codex |
| `multi_cross_db_vs_search` | Multiple | 3/9 | gpt-5.1, 5-codex, 5.1-codex |
| `irrel_slack_for_math` | Irrelevance | 1/9 | gpt-5-codex |
| `irrel_mem_for_sports` | Irrelevance | 1/9 | gpt-5.1-codex |
| `irrel_search_for_personal` | Irrelevance | 1/9 | gpt-5.1-codex |
| Others | Mixed | 1–2 | Various — see full list below |

### Irrelevance Failures (3 total)

Only `gpt-5-codex` and `gpt-5.1-codex` produced tool calls when they shouldn't have. All other models correctly abstained.

---

## Full Failure List

### Simple (16 failures)

| Scenario | Failed Models |
|----------|---------------|
| simple_db_update | ALL (9/9) |
| simple_calc | gpt-5, gpt-5.1, gpt-5.2, gpt-5.3-codex, gpt-5.3-codex-spark |
| simple_stats | gpt-5.1, gpt-5.3-codex-spark |

### Multiple (58 failures)

| Scenario | Failed Models |
|----------|---------------|
| multi_gh_issue_vs_pr | ALL (9/9) |
| multi_db_update_vs_delete | gpt-5, gpt-5.1, gpt-5.2, gpt-5-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex, gpt-5.1-codex-max |
| multi_convert_vs_calc | gpt-5, gpt-5.1, gpt-5.2, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex, gpt-5.1-codex-max |
| multi_gh_pr_vs_issue | gpt-5, gpt-5.1, gpt-5.2, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex |
| multi_db_insert_vs_others | gpt-5.1, gpt-5.2, gpt-5-codex, gpt-5.3-codex-spark, gpt-5.1-codex-max |
| multi_calc_vs_convert | gpt-5, gpt-5.1, gpt-5.2, gpt-5.3-codex, gpt-5.3-codex-spark |
| multi_fs_search_vs_list | gpt-5.2, gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-max |
| multi_email_send_vs_search | gpt-5, gpt-5.2, gpt-5.2-codex |
| multi_cross_db_vs_search | gpt-5.1, gpt-5-codex, gpt-5.1-codex |
| multi_stats_vs_calc | gpt-5, gpt-5.3-codex-spark |
| multi_docker_logs_vs_stop | gpt-5.2, gpt-5.1-codex |
| multi_fs_write_vs_read | gpt-5.2-codex |
| multi_db_describe_vs_query | gpt-5.3-codex-spark |

### Parallel (17 failures)

| Scenario | Failed Models |
|----------|---------------|
| para_calc_multi | gpt-5.1, gpt-5.2, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex, gpt-5.1-codex-max |
| para_fs_multi_read | gpt-5-codex, gpt-5.1-codex-max |
| para_db_multi_query | gpt-5.2, gpt-5.1-codex-max |
| para_convert_multi | gpt-5-codex, gpt-5.1-codex |
| para_fs_multi_stat | gpt-5-codex |
| para_db_multi_describe | gpt-5-codex |
| para_fs_multi_list | gpt-5-codex |
| para_email_multi_search | gpt-5.1-codex |

### Parallel Multiple (60 failures)

| Scenario | Failed Models |
|----------|---------------|
| pmulti_fs_rw | ALL (9/9) |
| pmulti_fs_db | ALL (9/9) |
| pmulti_docker_run_ps | gpt-5, gpt-5.1, gpt-5.2, gpt-5-codex, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex-max |
| pmulti_cal_create_list | gpt-5, gpt-5.1, gpt-5.2, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex-max |
| pmulti_3way_ops | gpt-5, gpt-5.2, gpt-5-codex, gpt-5.3-codex-spark, gpt-5.1-codex, gpt-5.1-codex-max |
| pmulti_gh_issue_list | gpt-5, gpt-5-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex |
| pmulti_mem_create | gpt-5, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex, gpt-5.1-codex-max |
| pmulti_db_slack | gpt-5.1, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.1-codex |
| pmulti_db_query_describe | gpt-5-codex, gpt-5.1-codex |
| pmulti_search_weather | gpt-5-codex |
| pmulti_slack_send_react | gpt-5-codex |
| pmulti_calc_convert | gpt-5-codex |
| pmulti_fs_list_search | gpt-5-codex |
| pmulti_gh_email | gpt-5.1-codex |

### Irrelevance (3 failures)

| Scenario | Failed Models |
|----------|---------------|
| irrel_slack_for_math | gpt-5-codex |
| irrel_mem_for_sports | gpt-5.1-codex |
| irrel_search_for_personal | gpt-5.1-codex |

---

## Key Findings

### 1. Parallel Multiple is the hardest category (60.8% pass rate)
Models strongly prefer sequential execution — calling one function, getting the result, then deciding the next. This is fundamentally a single-turn limitation: true multi-step orchestration requires multi-turn tool feedback loops.

### 2. Math/Conversion tool discrimination is weak
When both `calculate` and `convert_units` are available, models frequently pick the "wrong" one. Both are often valid approaches (e.g., `72°F to °C` can be done via `convert_units` OR `calculate((72-32)*5/9)`). This suggests the boundary between these tools needs clearer schema descriptions or the tests should accept either.

### 3. GitHub action selection shows "cautious" model behavior
For "report a bug," models prefer `list_issues` (check existing) over `create_issue` (immediately create). For "open a PR," models first want context. This is arguably *good* agent behavior but doesn't match the strict BFCL expectation.

### 4. Structured DB operations vs raw SQL
Models prefer `db_query` with raw SQL (`UPDATE ... SET ... WHERE ...`) over the structured `db_update(table, set, where)` format. The structured API decomposition doesn't match how models think about database operations.

### 5. gpt-5.2-codex is the most capable (91.0%)
Best across all categories. `gpt-5-codex` and `gpt-5.1-codex` are weakest, particularly in parallel and irrelevance detection.

### 6. Irrelevance detection is near-perfect (97.8%)
Only codex-era models (`gpt-5-codex`, `gpt-5.1-codex`) occasionally call tools when they shouldn't. All other models correctly abstain.

---

## Recommendations

1. **Relax `simple_db_update`**: Accept if model provides `table` + `where` containing SET clause, even without separate `set` arg.
2. **Relax `multi_gh_issue_vs_pr`**: Accept `list_issues` as valid first step (cautious agent pattern).
3. **Accept alternative math tools**: For `multi_calc_vs_convert` / `multi_convert_vs_calc`, accept either `calculate` or `convert_units`.
4. **Relax Parallel Multiple min_calls**: For read+write type scenarios, accept 1 call (the read) as partial success.
5. **Add multi-turn variant**: True sequential tool-call testing requires tool-result feedback loops.

---

## Environment

- **Python**: 3.13.12
- **pytest**: 9.0.2
- **GPTMock**: in-process via `TestClient`
- **Upstream**: OpenAI API (proxied through GPTMock model aliases)
- **9 Models**: gpt-5, gpt-5.1, gpt-5.2, gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark
