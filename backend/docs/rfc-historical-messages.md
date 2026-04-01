# RFC: Historical Message Storage

## 1. Problem

DeerFlow currently uses LangGraph's **Checkpointer** as the sole persistence layer for agent state. This creates three structural limitations:

### 1.1 Single-State Overwrite

The Checkpointer stores the *current* state of a thread. Each new run updates the same checkpoint. There is no built-in mechanism to distinguish or retrieve the state from a *previous* run within the same `thread_id`.

### 1.2 Context Compression Destroys History

`SummarizationMiddleware` fires when the message list exceeds a token threshold. It replaces a window of older messages with a single summary `AIMessage`. After compression, the original messages are permanently gone from the checkpoint. The frontend therefore shows a truncated or summarized conversation history.

### 1.3 No Structured Run Audit Trail

There is currently no record of:
- Which runs happened under a given thread (start time, end time, model, status).
- The raw, uncompressed message stream that was sent/received during a run.
- User feedback (like/dislike) associated with a specific run.
- Which follow-up messages belong to which originating run.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Record every run under a thread with full lifecycle metadata. |
| G2 | Persist the complete, uncompressed message stream for every run. |
| G3 | Allow the frontend to display the full conversation history across multiple runs. |
| G4 | Support user feedback (thumbs-up/down) scoped to a run. |
| G5 | Associate follow-up messages with the run that produced the content they follow. |
| G6 | Support a "My Content" view listing all files produced across all threads. |

---

## 3. Non-Goals

- Replacing the LangGraph Checkpointer for agent state management.
- Full-text search across message content.
- Real-time sync across multiple server instances (future work).

---

## 4. Design

### 4.1 Runs Table

Introduce a `runs` record, stored in the same LangGraph **Store** backend (memory / SQLite / PostgreSQL) that already backs thread metadata.

**Namespace**: `("runs",)`

**Run record schema**:

```python
@dataclass
class RunRecord:
    run_id: str               # UUID4, matches the LangGraph run_id
    thread_id: str            # Parent thread
    status: str               # "pending" | "running" | "success" | "error" | "timeout" | "interrupted"
    created_at: str           # ISO-8601 UTC
    updated_at: str           # ISO-8601 UTC
    model_name: str | None    # Model used for this run
    metadata: dict            # Arbitrary extra data (agent_name, etc.)
    journal_path: str | None  # Relative path to the JSONL journal file
    last_ai_message: str | None  # Text of the final AI message (convenience field)
```

The runs table is accessed through the same `store.aget` / `store.aput` interface:

```python
RUNS_NS = ("runs",)  # store namespace

async def get_run(store, run_id: str) -> RunRecord | None: ...
async def put_run(store, record: RunRecord) -> None: ...
async def list_runs_for_thread(store, thread_id: str) -> list[RunRecord]: ...
```

Listing runs for a thread uses `store.asearch(RUNS_NS, filter={"thread_id": thread_id})`.

---

### 4.2 RunJournal — LangChain Callback Handler

`RunJournal` is a `BaseCallbackHandler` that captures every message event during a run and appends it to a JSONL file. It uses the same event shape that the stream API already emits to the frontend (SSE `custom` events), ensuring the journal can be replayed as-is.

```python
# backend/packages/harness/deerflow/agents/journal.py

from langchain_core.callbacks import BaseCallbackHandler

class RunJournal(BaseCallbackHandler):
    """
    Records all agent events during a run to a JSONL file.

    Each line is a JSON object with the same schema as the custom SSE events
    sent to the frontend:
        {"event": "<event_type>", "data": {...}, "ts": "<ISO timestamp>"}
    """

    def __init__(self, run_id: str, thread_id: str, journal_path: Path):
        self.run_id = run_id
        self.thread_id = thread_id
        self.journal_path = journal_path
        self._file: IO | None = None

    # --- Lifecycle ---

    def on_chain_start(self, ...): self._open(); self._write("run_start", {...})
    def on_chain_end(self, ...):   self._write("run_end", {...}); self._close()
    def on_chain_error(self, ...): self._write("run_error", {...}); self._close()

    # --- Message events (mirror stream API custom event types) ---

    def on_llm_start(self, ...):          self._write("llm_start", {...})
    def on_llm_new_token(self, ...):      self._write("llm_token", {...})
    def on_llm_end(self, ...):            self._write("llm_end", {...})
    def on_tool_start(self, ...):         self._write("tool_start", {...})
    def on_tool_end(self, ...):           self._write("tool_end", {...})
    def on_tool_error(self, ...):         self._write("tool_error", {...})
    def on_custom_event(self, name, data, ...):
        # Captures all custom events already dispatched by the agent
        # (task_running, task_finish, summarization, clarification, etc.)
        self._write(name, data)

    # --- Helpers ---

    def _write(self, event: str, data: dict) -> None:
        line = json.dumps({"event": event, "data": data, "ts": utc_now()})
        self._file.write(line + "\n")
        self._file.flush()
```

**Attachment**: `RunJournal` is registered as a callback when the agent graph is invoked:

```python
# Inside the run execution path (e.g., thread_runs router)
journal = RunJournal(run_id=run_id, thread_id=thread_id, journal_path=journal_path)
config["callbacks"] = [journal]
await graph.astream(input, config=config, ...)
```

---

### 4.3 Journal File Storage

#### Local storage (default)

```
.deer-flow/
└── threads/
    └── {thread_id}/
        └── runs/
            ├── {run_id_1}.jsonl
            ├── {run_id_2}.jsonl
            └── {run_id_3}.jsonl
```

The `.deer-flow/threads/{thread_id}/` directory is already created by `ThreadDataMiddleware` for workspace, uploads, and outputs. The `runs/` subdirectory is created by `RunJournal` on first write.

#### S3 / object storage (optional, future)

When object storage is configured, the journal path stored in the run record becomes an S3 URI:

```
s3://{bucket}/threads/{thread_id}/runs/{run_id}.jsonl
```

The `RunJournal` writer is abstracted behind a `JournalWriter` protocol to support both local and S3 backends without changing the callback logic.

#### JSONL event format

Each line in the journal is a self-contained JSON object:

```jsonl
{"event":"run_start","data":{"run_id":"...","thread_id":"...","model":"gpt-4o","input":{...}},"ts":"2026-01-01T00:00:00Z"}
{"event":"llm_start","data":{"prompt_tokens":1234},"ts":"2026-01-01T00:00:01Z"}
{"event":"llm_token","data":{"content":"Hello"},"ts":"2026-01-01T00:00:01.100Z"}
{"event":"tool_start","data":{"type":"tool_call","tool_name":"web_search","tool_call_id":"...","args":{...}},"ts":"2026-01-01T00:00:02Z"}
{"event":"tool_end","data":{"type":"tool_result","tool_call_id":"...","result":"..."},"ts":"2026-01-01T00:00:03Z"}
{"event":"task_running","data":{...},"ts":"..."}
{"event":"task_finish","data":{...},"ts":"..."}
{"event":"summarization","data":{"replaced_message_ids":[...],"summary":"..."},"ts":"..."}
{"event":"run_end","data":{"status":"success","output_message":"..."},"ts":"2026-01-01T00:00:10Z"}
```

The `summarization` event is critical: it records which messages were compressed and the resulting summary text, enabling faithful reconstruction and frontend display control.

---

### 4.4 New Gateway API Endpoints

```
GET  /api/threads/{thread_id}/runs
     → list[RunRecord]  (ordered by created_at desc)

GET  /api/threads/{thread_id}/runs/{run_id}/journal
     → streaming JSONL or paginated JSON array of journal events

GET  /api/threads/{thread_id}/runs/{run_id}/messages
     → reconstructed message list from the journal (convenience endpoint)
```

The `/messages` endpoint replays the journal, filtering to only human and final AI messages, and returns them in the same format as the existing thread state `values.messages` field — so the frontend can use the same rendering pipeline.

---

### 4.5 Frontend Changes

#### 4.5.1 Historical Messages State

Introduce a `historyMessages` state slice alongside the existing `messages` state:

```typescript
interface ThreadChatState {
  // Current run messages (live streaming or latest run)
  messages: Message[];

  // Messages from previous runs, loaded on demand
  historyMessages: HistoricalRun[];

  // Run metadata list (used for pagination)
  runs: RunSummary[];
  hasMoreRuns: boolean;
}

interface HistoricalRun {
  runId: string;
  createdAt: string;         // Shown as a separator between runs
  messages: Message[];
  isSummarized: boolean;     // True if messages were compressed mid-run
}
```

On thread load, the frontend fetches the latest run's journal via `/api/threads/{thread_id}/runs/{run_id}/messages` and populates `messages`.

#### 4.5.2 Summarization Event Handling

When a `summarization` event arrives in the SSE stream (dispatched by `SummarizationMiddleware`):

1. The current `messages` are moved to `historyMessages` as a new `HistoricalRun` with `isSummarized: true`.
2. The summarized `AIMessage` (with a distinct `name` marker, e.g., `name: "summary"`) is **not** displayed in the main message list by default.
3. A collapsed indicator — *"N messages summarized — click to expand"* — is rendered in place of the removed messages.

This prevents the jarring experience of messages disappearing and gives users an opt-in path to see the full history.

```typescript
// In useThreadStream, when a summarization event is received:
case "summarization": {
  dispatch({
    type: "SUMMARIZATION_RECEIVED",
    payload: {
      replacedMessageIds: event.data.replaced_message_ids,
      summary: event.data.summary,
    },
  });
  break;
}
```

#### 4.5.3 Infinite Scroll — Load Previous Runs

The message list supports upward scroll-to-load. When the user scrolls to the top:

1. The frontend calls `GET /api/threads/{thread_id}/runs?before={oldest_run_id}`.
2. The previous run's messages are prepended to `historyMessages`.
3. A **system timestamp message** is injected as a visual separator between runs:

```
─────────────  Mon, Jan 1 2026 · 14:32  ─────────────
```

Each run separator also shows the model name and status (success/error) as a lightweight audit trail.

---

### 4.6 User Feedback

Introduce a `feedback` record, also stored in the LangGraph Store:

**Namespace**: `("feedback",)`

```python
@dataclass
class FeedbackRecord:
    feedback_id: str      # UUID4
    run_id: str           # Associated run
    thread_id: str        # Denormalized for efficient lookup
    user_id: str | None   # From auth context
    rating: int           # +1 (thumbs-up) or -1 (thumbs-down)
    comment: str | None   # Optional free-text
    created_at: str       # ISO-8601 UTC
    message_id: str | None  # If feedback is scoped to a specific message
```

**New endpoints**:

```
POST /api/threads/{thread_id}/runs/{run_id}/feedback
     body: { "rating": 1 | -1, "comment": "...", "message_id": "..." }
     → FeedbackRecord

GET  /api/threads/{thread_id}/runs/{run_id}/feedback
     → list[FeedbackRecord]
```

The `{run_id}.jsonl` journal provides the full execution trace for any follow-up analysis or quality evaluation tied to feedback.

---

### 4.7 Follow-up Message Association

When the user submits a follow-up message in a thread, the originating `run_id` is recorded in the message metadata:

```python
# In the run input construction (thread_runs router)
input_message = HumanMessage(
    content=user_text,
    metadata={
        "follow_up_to_run_id": previous_run_id,  # nullable
        "message_id": str(uuid4()),
    }
)
```

This allows the system to trace which prior run a follow-up question refers to, which is useful for:
- Contextual memory retrieval.
- Feedback attribution when a user rates a follow-up answer.
- Journal replay that can link follow-ups back to the original content.

---

### 4.8 "My Content" Menu

Add a new top-level navigation item *"My Content"* that aggregates files generated across all threads for the current user.

**Backend endpoint**:

```
GET /api/users/me/outputs
    → list[OutputFile]

@dataclass
class OutputFile:
    thread_id: str
    thread_title: str
    run_id: str
    file_path: str       # Relative path under .deer-flow/threads/{thread_id}/outputs/
    file_name: str
    created_at: str
```

The endpoint queries the Store for all threads owned by the user, then enumerates the `outputs/` directory of each thread (already tracked by `ThreadDataMiddleware` via `ThreadDataState.outputs_dir`).

**Frontend file tree**:

```
My Content
├── Thread: "Quarterly Report Analysis"   [2 files]
│   ├── report_summary.md
│   └── data_chart.png
├── Thread: "Code Review Assistant"        [1 file]
│   └── review_notes.txt
└── Thread: "Research on LLMs"             [3 files]
    ├── literature_review.pdf
    ├── comparison_table.csv
    └── notes.md
```

Only files in the `outputs/` directory (i.e., files explicitly written by the agent for the user) are shown — not internal workspace files or uploaded inputs.

---

## 5. Data Flow

```
User Request
    │
    ▼
Gateway: POST /api/threads/{thread_id}/runs/stream
    │
    ├─ Create RunRecord (status=pending) → Store
    ├─ Resolve journal_path
    ├─ Instantiate RunJournal(run_id, thread_id, journal_path)
    └─ Invoke LangGraph graph with callbacks=[journal]
            │
            ├─ RunJournal.on_chain_start()  → JSONL: run_start
            ├─ RunJournal.on_llm_start()   → JSONL: llm_start
            ├─ RunJournal.on_llm_new_token() → JSONL: llm_token (streamed)
            ├─ RunJournal.on_tool_start()  → JSONL: tool_start
            ├─ RunJournal.on_tool_end()    → JSONL: tool_end
            ├─ RunJournal.on_custom_event("summarization", ...) → JSONL: summarization
            └─ RunJournal.on_chain_end()   → JSONL: run_end
    │
    ├─ Update RunRecord (status=success, last_ai_message=...) → Store
    └─ SSE stream to frontend (same events, sourced from LangGraph)

Frontend
    ├─ Renders live messages from SSE stream
    ├─ On "summarization" event → moves messages to historyMessages
    ├─ On scroll-to-top → GET /runs?before={id} → prepend historical run
    └─ On feedback submit → POST /runs/{id}/feedback
```

---

## 6. Storage Layout Summary

```
LangGraph Store (memory / SQLite / PostgreSQL)
├── namespace ("threads",)
│   └── {thread_id} → ThreadRecord
├── namespace ("runs",)
│   └── {run_id}    → RunRecord  (contains journal_path)
└── namespace ("feedback",)
    └── {feedback_id} → FeedbackRecord

File system (local) or Object Storage (S3)
└── .deer-flow/
    └── threads/
        └── {thread_id}/
            ├── workspace/       (existing)
            ├── uploads/         (existing)
            ├── outputs/         (existing)
            └── runs/            (NEW)
                ├── {run_id_1}.jsonl
                └── {run_id_2}.jsonl
```

---

## 7. Rollout Phases

### Phase 1 — Backend Journal (no breaking changes)

1. Create `RunJournal(BaseCallbackHandler)` in `deerflow/agents/journal.py`.
2. Add `runs/` directory creation to `ThreadDataMiddleware`.
3. Inject `RunJournal` into the run config in the thread_runs router.
4. Persist `RunRecord` to Store on run start/end.
5. Add `GET /api/threads/{thread_id}/runs` and `GET /api/threads/{thread_id}/runs/{run_id}/messages` endpoints.
6. Emit a `summarization` custom event from `SummarizationMiddleware` with `replaced_message_ids`.

### Phase 2 — Frontend History UX

1. Add `historyMessages` and `runs` to thread chat state.
2. Handle `summarization` SSE event: move messages, show collapsed indicator.
3. Load latest run messages from `/runs/{run_id}/messages` on thread open.
4. Implement scroll-to-load-previous-run with run separator component.

### Phase 3 — Feedback & Follow-ups

1. Add `FeedbackRecord` store + feedback API endpoints.
2. Add thumbs-up / thumbs-down UI to message groups.
3. Record `follow_up_to_run_id` in human message metadata.

### Phase 4 — My Content

1. Add `GET /api/users/me/outputs` endpoint.
2. Add "My Content" navigation item and file tree component to the frontend.

---

## 8. Open Questions

| # | Question | Notes |
|---|----------|-------|
| Q1 | Should `RunJournal` write synchronously or buffer async? | Sync is simpler and avoids dropped events on crash. |
| Q2 | Should the JSONL journal be compressed (gzip)? | Useful for long-running runs; can be toggled per config. |
| Q3 | Token-level streaming (`llm_token`) produces very large journals. Should it be opt-in? | Default: off. Enable with `journal.stream_tokens = True` in config. |
| Q4 | How are runs scoped per user in a multi-user deployment? | The Store already supports user-scoped namespaces; runs can follow the same pattern. |
| Q5 | Is a `messages` table needed in addition to the journal? | Current proposal: store `last_ai_message` on the `RunRecord` as a convenience field; full messages live in the JSONL. A dedicated table can be added if query patterns demand it. |
| Q6 | Journal file retention policy? | Propose configurable TTL (default: no expiry). A cleanup job can archive or delete old journals. |
