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

import json
from langchain_core.callbacks import BaseCallbackHandler
from deerflow.runtime.serialization import serialize_lc_object

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
    # on_llm_new_token is intentionally omitted by default (see Q3)
    def on_llm_end(self, response, ...):  self._write("llm_end", {"message": serialize_lc_object(response.generations[0][0].message)})
    def on_tool_start(self, ...):         self._write("tool_start", {...})
    def on_tool_end(self, ...):           self._write("tool_end", {...})
    def on_tool_error(self, ...):         self._write("tool_error", {...})
    def on_custom_event(self, name, data, ...):
        # Captures all custom events already dispatched by the agent
        # (task_running, task_finish, summarization, clarification, etc.)
        self._write(name, data)

    # --- Helpers ---

    def _write(self, event: str, data: Any) -> None:
        line = json.dumps({"event": event, "data": serialize_lc_object(data), "ts": utc_now()})
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

#### 4.2.1 Serialization: Reuse `serialize_lc_object`

All journal writes must route through `deerflow.runtime.serialization.serialize_lc_object` — the same helper used by `worker.py` when publishing SSE events. This ensures:

- LangChain / LangGraph objects (`AIMessage`, `ToolMessage`, `HumanMessage`, `LLMResult`, …) are converted via `model_dump()` (Pydantic v2) or `dict()` (Pydantic v1) before `json.dumps`.
- Nested dicts and lists are recursively handled.
- Unknown objects fall back to `str()`.

Calling `json.dumps` directly on callback arguments (e.g., the `LLMResult` passed to `on_llm_end`) without this step will raise `TypeError: Object of type AIMessage is not JSON serializable`.

---

#### 4.2.2 Callback Completeness: No Manual Chunk Merging Required

LangGraph's `graph.astream(stream_mode="messages")` emits **`AIMessageChunk`** objects — one per streaming token. Accumulating them into a full `AIMessage` would require manual `reduce`/`+` logic in the callback.

The callback path avoids this entirely:

| Callback | Data received | Completeness |
|----------|--------------|--------------|
| `on_llm_new_token(token)` | Single `str` token | ❌ chunk — skip for journal (see Q3) |
| `on_llm_end(response)` | `LLMResult` with `generations[0][0].message` | ✅ **complete `AIMessage`** — use this |
| `on_tool_start(serialized, inputs)` | Full tool input dict | ✅ complete |
| `on_tool_end(output)` | Full tool output `str` | ✅ complete |
| `on_custom_event(name, data)` | Full custom event payload | ✅ complete |

**Decision**: `RunJournal` writes the complete `AIMessage` from `on_llm_end` into the `llm_end` journal entry. `on_llm_new_token` is **not** called by default (opt-in via `journal.stream_tokens = True`; see Open Question Q3).

---

#### 4.2.3 Format Compatibility: Callback Messages vs `useStream` Messages

The frontend's `useStream` hook (from `@langchain/langgraph-sdk/react`) populates `thread.messages` from the LangGraph `values`-mode SSE stream. In the deerflow gateway, `worker.py` serializes those values via `serialize_channel_values` → `serialize_lc_object` → `model_dump()`. The result is the same dict shape as calling `AIMessage.model_dump()` directly.

`AIMessage.model_dump()` produces:

```json
{
  "type": "ai",
  "content": "...",
  "id": "run-<uuid>-0",
  "name": null,
  "tool_calls": [],
  "invalid_tool_calls": [],
  "additional_kwargs": {},
  "response_metadata": {"model_name": "gpt-4o", ...},
  "usage_metadata": {...},
  "example": false
}
```

The TypeScript `Message` type from `@langchain/langgraph-sdk` uses the same field names (`type`, `content`, `id`, `tool_calls`, …). The extra Python-only fields (`example`, `invalid_tool_calls`) are ignored by the TypeScript SDK as unknown keys. **The structure is compatible**: a callback-serialized `AIMessage` can be served directly from the `/messages` endpoint and rendered by the existing frontend message components.

**One important caveat — the `id` field**:

- **From `values` stream (LangGraph state)**: The message `id` is assigned by LangGraph's state reducer for deduplication (e.g., `"run-<uuid>-0"`). This ID is stable and matches what `useStream` stores.
- **From `on_llm_end` callback**: The message `id` in the `LLMResult` comes from the LLM provider's response or is auto-generated by `langchain_core` at call time. It may be a different value (e.g., `"chatcmpl-xxx"`).

This means that if the frontend tries to correlate journal messages with live-stream messages by `id`, the IDs may not match. The `/messages` endpoint should therefore be treated as a **standalone reconstruction** for display purposes, not as a delta against an existing in-memory state. For the historical display use-case (loading previous runs that are no longer in the active stream), this is not a problem — the frontend renders the returned list as-is.

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

All frontend changes live in **`frontend/src/core/threads/hooks.ts`**, inside the `useThreadStream` hook. No new hooks or files are required for the core logic.

#### 4.5.1 Historical Messages State

Add three new `useState` entries at the top of `useThreadStream`, alongside the existing `optimisticMessages` and `isUploading` state:

```typescript
// frontend/src/core/threads/hooks.ts — inside useThreadStream()

// Messages from previous runs, loaded on demand (prepended on scroll-up)
const [historyMessages, setHistoryMessages] = useState<HistoricalRun[]>([]);
// Run summary list for the current thread (used for pagination)
const [runs, setRuns] = useState<RunSummary[]>([]);
const [hasMoreRuns, setHasMoreRuns] = useState(false);
```

Supporting types (add to `frontend/src/core/threads/types.ts`):

```typescript
export interface HistoricalRun {
  runId: string;
  createdAt: string;        // Used as a visual separator between runs
  messages: Message[];
  isSummarized: boolean;    // True if context compression fired mid-run
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  status: string;
  model_name: string | null;
}
```

On thread load (when `threadId` becomes defined), fetch the latest run's reconstructed messages and populate the current run's display. This replaces relying solely on the Checkpointer state, which may already be compressed:

```typescript
// frontend/src/core/threads/hooks.ts — add useEffect inside useThreadStream()

useEffect(() => {
  if (!threadId) return;
  void (async () => {
    const res = await fetch(
      `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/runs?limit=20`,
    );
    if (!res.ok) return;
    const data: RunSummary[] = await res.json();
    setRuns(data);
    setHasMoreRuns(data.length === 20);
  })();
}, [threadId]);
```

#### 4.5.2 Summarization Event Handling

The `summarization` custom event is dispatched by `SummarizationMiddleware` on the backend. Handle it inside the existing `onCustomEvent` callback that is already passed to `useStream`:

```typescript
// frontend/src/core/threads/hooks.ts — extend the onCustomEvent handler inside useStream()

onCustomEvent(event: unknown) {
  if (
    typeof event === "object" &&
    event !== null &&
    "type" in event
  ) {
    if (event.type === "task_running") {
      // existing logic …
      const e = event as { type: "task_running"; task_id: string; message: AIMessage };
      updateSubtask({ id: e.task_id, latestMessage: e.message });
    }

    // NEW: context compression fired — snapshot current messages into history
    if (event.type === "summarization") {
      const e = event as {
        type: "summarization";
        replaced_message_ids: string[];
        summary: string;
        run_id: string;
      };
      // Move the messages that are about to be replaced into historyMessages
      setHistoryMessages((prev) => [
        ...prev,
        {
          runId: e.run_id,
          createdAt: new Date().toISOString(),
          messages: thread.messages.filter((m) =>
            e.replaced_message_ids.includes(m.id ?? ""),
          ),
          isSummarized: true,
        },
      ]);
    }
  }
},
```

The summary `AIMessage` produced by the backend carries a distinct `name: "summary"` field. Callers that render messages (e.g., `message-list.tsx`) should check for this marker and render a collapsed indicator — *"N messages summarized — click to expand"* — rather than showing the raw summary prose by default.

#### 4.5.3 Load Previous Runs on Scroll-Up

Add a `loadPreviousRun` callback inside `useThreadStream`. The message list component calls this when the user scrolls to the top:

```typescript
// frontend/src/core/threads/hooks.ts — add inside useThreadStream()

const loadPreviousRun = useCallback(async () => {
  if (!threadId || !hasMoreRuns || runs.length === 0) return;
  const oldest = runs[runs.length - 1];
  if (!oldest) return;

  const res = await fetch(
    `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(oldest.run_id)}/messages`,
  );
  if (!res.ok) return;
  const messages: Message[] = await res.json();

  setHistoryMessages((prev) => [
    {
      runId: oldest.run_id,
      createdAt: oldest.created_at,
      messages,
      isSummarized: false,
    },
    ...prev,
  ]);

  // Fetch the next page of run summaries
  const nextRes = await fetch(
    `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/runs?before=${encodeURIComponent(oldest.run_id)}&limit=20`,
  );
  if (nextRes.ok) {
    const nextRuns: RunSummary[] = await nextRes.json();
    setRuns((prev) => [...prev, ...nextRuns]);
    setHasMoreRuns(nextRuns.length === 20);
  }
}, [threadId, hasMoreRuns, runs]);
```

A **system timestamp message** is injected as a visual separator between runs by the rendering layer (not the hook). Each separator shows the run's `createdAt` timestamp, model name, and status:

```
─────────────  Mon, Jan 1 2026 · 14:32  ─────────────
```

#### 4.5.4 Updated Return Value

Extend the hook's return tuple to expose the new state and callback:

```typescript
// frontend/src/core/threads/hooks.ts — update the return at the end of useThreadStream()

return [
  mergedThread,
  sendMessage,
  isUploading,
  historyMessages,
  loadPreviousRun,
  hasMoreRuns,
] as const;
```

Update the corresponding destructuring at every call site (currently `use-thread-chat.ts`).

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
            ├─ RunJournal.on_llm_new_token() → JSONL: llm_token (opt-in, default off)
            ├─ RunJournal.on_llm_end()     → JSONL: llm_end (complete AIMessage via serialize_lc_object)
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
| Q3 | Token-level streaming (`llm_token`) produces very large journals. Should it be opt-in? | Default: off. `on_llm_new_token` is not implemented by default. Enable with `journal.stream_tokens = True` in config. The complete message is always recorded via `on_llm_end`. |
| Q4 | How are runs scoped per user in a multi-user deployment? | The Store already supports user-scoped namespaces; runs can follow the same pattern. |
| Q5 | Is a `messages` table needed in addition to the journal? | Current proposal: store `last_ai_message` on the `RunRecord` as a convenience field; full messages live in the JSONL. A dedicated table can be added if query patterns demand it. |
| Q6 | Journal file retention policy? | Propose configurable TTL (default: no expiry). A cleanup job can archive or delete old journals. |
| Q7 | The `id` field on callback-sourced messages differs from LangGraph state message IDs. Does this matter? | The `/messages` endpoint is a standalone reconstruction for display only — it is never delta-merged with live-stream state. The `id` mismatch is therefore not a problem for the historical view use-case (see §4.2.3). |
| Q8 | Should `_write` use `serialize_lc_object` or a custom encoder? | Use `serialize_lc_object` from `deerflow.runtime.serialization` — same helper used by `worker.py` — to keep the JSONL format consistent with the SSE payload shape. |
