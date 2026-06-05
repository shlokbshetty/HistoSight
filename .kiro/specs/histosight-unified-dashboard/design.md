# Design Document: HistoSight Unified Dashboard

## Overview

This document describes the technical design for the HistoSight Unified Dashboard feature. The work spans three layers:

1. **Python training fixes** — Repair the broken `nt_xent_loss` and `contrastive_top1_acc` functions in `train.py` and reduce wall-clock training time by tuning `MAX_TRAIN_SAMPLES` and `TOTAL_EPOCHS`.
2. **FastAPI backend extension** — Add `POST /train_baseline` and `GET /train_baseline/status` endpoints to `backend/main.py` that manage a supervised baseline training subprocess.
3. **Frontend SPA** — Produce a single `index.html` at the project root that unifies the four HTML fragments from `stitch_frontend_prompt.md` into a five-section SPA (Upload, Analytics, Comparison, Test, About) with sidebar navigation, Chart.js charts, 30-second polling, and auto-trigger of the baseline training job.

No new frameworks or build tools are introduced. All Python changes are confined to `train.py` and `backend/main.py`. The frontend is pure vanilla JS + Tailwind CDN + Chart.js CDN.

---

## Architecture

```
project root/
├── index.html                  ← NEW: unified SPA (vanilla JS, Tailwind CDN, Chart.js CDN)
├── train.py                    ← MODIFIED: fix nt_xent_loss, contrastive_top1_acc, reduce epochs/samples
├── backend/
│   └── main.py                 ← MODIFIED: add POST /train_baseline, GET /train_baseline/status
├── train_supervised_baseline.py ← UNCHANGED (spawned as subprocess)
├── inference.py                ← UNCHANGED (do not modify)
├── logs/
│   ├── metrics.json            ← written by train.py, read by GET /metrics
│   └── comparison.json         ← written by train.py + train_supervised_baseline.py, read by GET /comparison
└── checkpoints/ models/        ← UNCHANGED
```

### Request Flow

```
Browser (index.html)
  │
  ├─ DOMContentLoaded → POST /train_baseline  ──► backend/main.py
  │                                                  └─ subprocess.Popen(train_supervised_baseline.py)
  │
  ├─ setInterval(30s) → GET /metrics          ──► backend/main.py → logs/metrics.json
  │                   → GET /comparison       ──► backend/main.py → logs/comparison.json
  │                   → GET /train_baseline/status
  │
  └─ Upload → POST /predict                   ──► backend/main.py → inference.py
```

---

## Components and Interfaces

### 1. `train.py` — NT-Xent Loss Fix

**`nt_xent_loss(z1, z2, temperature=TEMPERATURE)`**

- Inputs: `z1`, `z2` — L2-normalized tensors of shape `(N, D)`; `temperature` — positive float scalar
- Output: scalar `torch.Tensor` (cross-entropy loss)
- Algorithm:
  ```python
  z = torch.cat([z1, z2], dim=0)          # (2N, D)
  N = z1.shape[0]
  sim = (z @ z.T) / temperature            # (2N, 2N)
  mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
  sim = sim.masked_fill(mask, -1e4)
  labels = torch.cat([
      torch.arange(N, 2*N, device=z.device),
      torch.arange(0, N,   device=z.device),
  ])
  return F.cross_entropy(sim, labels)
  ```

**`contrastive_top1_acc(z1, z2, temperature=TEMPERATURE)`**

- Inputs: same as `nt_xent_loss` (detached tensors)
- Output: Python `float` in `[0.0, 1.0]`
- Algorithm: identical `z/N/sim/mask/labels` construction, then:
  ```python
  return (sim.argmax(dim=1) == labels).float().mean().item()
  ```

### 2. `train.py` — Hyperparameter Reduction

| Constant | Old Value | New Value |
|---|---|---|
| `MAX_TRAIN_SAMPLES` | 50000 | 40000 |
| `TOTAL_EPOCHS` | 55 | 45 |

All other constants (`PATIENCE`, `BATCH_SIZES`, `LR`, `TEMPERATURE`, etc.) remain unchanged.

### 3. `backend/main.py` — New Endpoints

**State management** (module-level globals):
```python
_baseline_proc: Optional[subprocess.Popen] = None
_baseline_done: bool = False
```

**`POST /train_baseline`** — Returns HTTP 202 `{"status": "started"}`, HTTP 200 `{"status": "already_running"}`, or HTTP 200 `{"status": "already_done"}`.

Guard logic (in order):
1. Check `comparison.json` for `resnet_supervised.accuracy > 0` → `already_done`
2. Check `_baseline_proc is not None and _baseline_proc.poll() is None` → `already_running`
3. Otherwise: `subprocess.Popen([sys.executable, baseline_script_path])`, return `started`

**`GET /train_baseline/status`** — Returns:
```json
{
  "status": "idle" | "running" | "done",
  "resnet_supervised": { "accuracy": float, "training_time": int, "epochs": int }
}
```

Status derivation:
- `done` if `comparison.json` has `resnet_supervised.accuracy > 0`
- `running` if `_baseline_proc` is not None and `poll() is None`
- `idle` otherwise

### 4. `index.html` — SPA Structure

**Sections** (each a `<section id="section-{name}">` div):
- `upload` — drag-drop zone, preview, predict button, result card
- `analytics` — 3 Chart.js line charts + stat cards
- `comparison` — 1 Chart.js grouped bar chart + summary table
- `test` — read-only card with edit/save via localStorage
- `about` — static project metadata

**Navigation** — `<aside>` sidebar with 5 `<a data-section="{name}">` links. Click handler:
```js
function navigate(name) {
  document.querySelectorAll('[data-section]').forEach(a => a.classList.remove('active'));
  document.querySelectorAll('section[id^="section-"]').forEach(s => s.classList.add('hidden'));
  document.querySelector(`[data-section="${name}"]`).classList.add('active');
  document.getElementById(`section-${name}`).classList.remove('hidden');
  if (name === 'analytics') fetchMetrics();
  if (name === 'comparison') fetchComparison();
}
```

**Polling** — `setInterval` at 30 000 ms, fetches `/metrics` and `/comparison` concurrently via `Promise.all`. Stops when `/train_baseline/status` returns `"done"`.

**Auto-trigger** — On `DOMContentLoaded`:
```js
fetch('/train_baseline', { method: 'POST' })
  .then(r => r.json())
  .then(d => console.log('[train_baseline]', d.status))
  .catch(e => showToast('Baseline training unavailable'));
```

**Chart instances** — Module-level variables `lossChart`, `accChart`, `valLossChart`, `compChart` initialized once and updated via `chart.data.datasets[i].data = newData; chart.update()`.

---

## Data Models

### `logs/metrics.json`
```json
{
  "epoch":          [1, 2, ...],
  "train_loss":     [2.31, ...],
  "val_loss":       [2.48, ...],
  "train_accuracy": [0.03, ...],
  "val_accuracy":   [0.03, ...],
  "epoch_time":     [312.4, ...]
}
```

### `logs/comparison.json`
```json
{
  "resnet_supervised": { "accuracy": 0.0, "training_time": 0, "epochs": 0 },
  "simclr_no_macenko": { "accuracy": 0.0, "training_time": 312.4, "epochs": 1 },
  "final_model":       { "accuracy": 0.0, "training_time": 0, "epochs": 0 }
}
```

### `localStorage` key: `histosight_test_results`
```json
{
  "accuracy": 0.8712,
  "loss": 0.3241,
  "inference_time": 124
}
```

### Backend subprocess state (in-memory)
```python
_baseline_proc: Optional[subprocess.Popen]  # None = never started
_baseline_done: bool                         # True = comparison.json has non-zero accuracy
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: NT-Xent loss is bounded above for perfect positive pairs

*For any* batch size N ≥ 2 and any L2-normalized projection tensor z, when `z1 == z2 == z`, the NT-Xent loss SHALL be strictly less than `log(2N - 1)` (the maximum possible cross-entropy loss when the positive pair is ranked last among all negatives).

**Validates: Requirements 1.6**

### Property 2: Contrastive top-1 accuracy is in [0.0, 1.0]

*For any* pair of L2-normalized tensors `z1` and `z2` of shape `(N, D)` with N ≥ 2, `contrastive_top1_acc(z1, z2)` SHALL return a Python float in the closed interval `[0.0, 1.0]`.

**Validates: Requirements 2.4**

### Property 3: Perfect positive pairs yield accuracy 1.0 and minimal loss

*For any* L2-normalized tensor z of shape `(N, D)` with N ≥ 2, when `z1 == z2 == z`, `contrastive_top1_acc(z1, z2)` SHALL return `1.0` and `nt_xent_loss(z1, z2)` SHALL return a value strictly less than `log(2N - 1)`. This consolidates the consistency requirement between loss and accuracy.

**Validates: Requirements 1.6, 2.5**

### Property 4: Navigation shows exactly one section

*For any* of the five section names (upload, analytics, comparison, test, about), after calling `navigate(name)`, exactly one `<section>` element SHALL be visible (not hidden) and the remaining four SHALL be hidden.

**Validates: Requirements 4.3**

### Property 5: Navigation applies active style to exactly one link

*For any* of the five section names, after calling `navigate(name)`, exactly one sidebar link SHALL carry the active CSS class and the remaining four SHALL not.

**Validates: Requirements 4.4**

### Property 6: Polling fires every 30 seconds

*For any* number of elapsed 30-second intervals k ≥ 1, the polling handler SHALL have been invoked exactly k times, fetching both `/metrics` and `/comparison` on each invocation.

**Validates: Requirements 6.4, 7.4, 9.1**

### Property 7: Polling stops when status is "done"

*For any* polling state, when `GET /train_baseline/status` returns `{"status": "done"}`, the polling interval SHALL be cleared and no further automatic fetches SHALL occur.

**Validates: Requirements 9.5**

### Property 8: Test results localStorage round-trip

*For any* valid test result object with `accuracy ∈ [0, 1]`, `loss ≥ 0`, and `inference_time ≥ 0`, saving via the Save button and then reloading the Test Section SHALL display the same values that were entered.

**Validates: Requirements 10.4**

### Property 9: Invalid accuracy is rejected

*For any* value outside the range `[0, 1]` (including negative numbers, values > 1, NaN, and non-numeric strings) entered as test accuracy, the Save button SHALL be disabled or the save action SHALL be prevented, and `localStorage` SHALL NOT be updated.

**Validates: Requirements 10.5**

---

## Error Handling

| Scenario | Component | Behavior |
|---|---|---|
| `POST /predict` returns HTTP error | Upload section | Inline error banner with message; spinner hidden |
| No file selected | Upload section | "Run Prediction" button disabled; placeholder state shown |
| `GET /metrics` returns 404 or network error | Analytics section | "No training data available" banner; charts not rendered |
| `GET /comparison` returns 404 or network error | Comparison section | "No comparison data available" banner |
| All comparison accuracies are 0.0 | Comparison section | "In Progress — baseline training running" banner alongside empty chart |
| `POST /train_baseline` network error | Dashboard startup | Silent `console.warn`; non-intrusive toast; UI continues loading |
| Polling fetch fails | Polling handler | `console.error`; interval continues; no blocking UI error |
| `nt_xent_loss` receives NaN/Inf inputs | `train.py` | Values propagate naturally; training loop's gradient clipping handles instability |
| `subprocess.Popen` fails in `/train_baseline` | Backend | HTTP 500 with descriptive error message |
| `comparison.json` missing when `/train_baseline/status` called | Backend | Returns `{"status": "idle", "resnet_supervised": {"accuracy": 0.0, "training_time": 0, "epochs": 0}}` |

---

## Testing Strategy

### Unit Tests (Python)

**`test_train_functions.py`** — Tests for `nt_xent_loss` and `contrastive_top1_acc`:
- Example: call with valid `(4, 128)` tensors, verify scalar output
- Example: call `contrastive_top1_acc` with valid inputs, verify float in `[0, 1]`
- Edge case: NaN input propagates to NaN output (not a finite value)
- Property tests: see below

**`test_backend_baseline.py`** — Tests for new backend endpoints using FastAPI `TestClient`:
- Example: `POST /train_baseline` with no prior state → 202 `started`
- Example: `POST /train_baseline` while process running → 200 `already_running`
- Example: `POST /train_baseline` with non-zero accuracy in `comparison.json` → 200 `already_done`
- Example: `GET /train_baseline/status` returns correct schema
- Example: existing endpoints (`/predict`, `/metrics`, `/comparison`) still respond correctly

### Property-Based Tests (Python — Hypothesis)

Property-based testing is appropriate here because `nt_xent_loss` and `contrastive_top1_acc` are pure functions with clear input/output behavior, universal properties that hold across all valid batch sizes and embedding dimensions, and cheap to run 100+ iterations (CPU tensors, small N).

**Library**: [Hypothesis](https://hypothesis.readthedocs.io/) (`pip install hypothesis`)

**`test_train_properties.py`**:

```python
# Feature: histosight-unified-dashboard, Property 1: NT-Xent loss bounded above for perfect pairs
@given(
    N=st.integers(min_value=2, max_value=32),
    D=st.integers(min_value=8, max_value=128),
)
@settings(max_examples=100)
def test_nt_xent_loss_bounded_perfect_pairs(N, D):
    z = F.normalize(torch.randn(N, D), dim=1)
    loss = nt_xent_loss(z, z)
    import math
    assert loss.item() < math.log(2 * N - 1)
```

```python
# Feature: histosight-unified-dashboard, Property 2: Top-1 accuracy in [0, 1]
@given(
    N=st.integers(min_value=2, max_value=32),
    D=st.integers(min_value=8, max_value=128),
)
@settings(max_examples=100)
def test_contrastive_acc_range(N, D):
    z1 = F.normalize(torch.randn(N, D), dim=1)
    z2 = F.normalize(torch.randn(N, D), dim=1)
    acc = contrastive_top1_acc(z1, z2)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
```

```python
# Feature: histosight-unified-dashboard, Property 3: Perfect pairs yield acc=1.0 and minimal loss
@given(
    N=st.integers(min_value=2, max_value=32),
    D=st.integers(min_value=8, max_value=128),
)
@settings(max_examples=100)
def test_perfect_pairs_consistency(N, D):
    import math
    z = F.normalize(torch.randn(N, D), dim=1)
    acc = contrastive_top1_acc(z, z)
    loss = nt_xent_loss(z, z)
    assert acc == 1.0
    assert loss.item() < math.log(2 * N - 1)
```

### Frontend Tests (JavaScript — Jest + jsdom)

Property-based testing for the frontend uses [fast-check](https://fast-check.dev/).

**`test_navigation.js`**:

```js
// Feature: histosight-unified-dashboard, Property 4: Navigation shows exactly one section
fc.assert(fc.property(
  fc.constantFrom('upload', 'analytics', 'comparison', 'test', 'about'),
  (name) => {
    navigate(name);
    const visible = [...document.querySelectorAll('section[id^="section-"]')]
      .filter(s => !s.classList.contains('hidden'));
    return visible.length === 1 && visible[0].id === `section-${name}`;
  }
), { numRuns: 100 });

// Feature: histosight-unified-dashboard, Property 5: Active style on exactly one link
fc.assert(fc.property(
  fc.constantFrom('upload', 'analytics', 'comparison', 'test', 'about'),
  (name) => {
    navigate(name);
    const active = [...document.querySelectorAll('[data-section]')]
      .filter(a => a.classList.contains('active'));
    return active.length === 1 && active[0].dataset.section === name;
  }
), { numRuns: 100 });
```

**`test_polling.js`**:

```js
// Feature: histosight-unified-dashboard, Property 6: Polling fires every 30 seconds
// Feature: histosight-unified-dashboard, Property 7: Polling stops when status is "done"
// Use jest fake timers + mock fetch
```

**`test_test_section.js`**:

```js
// Feature: histosight-unified-dashboard, Property 8: localStorage round-trip
fc.assert(fc.property(
  fc.float({ min: 0, max: 1 }),
  fc.float({ min: 0, max: 10 }),
  fc.integer({ min: 0, max: 10000 }),
  (accuracy, loss, inferenceTime) => {
    saveTestResults(accuracy, loss, inferenceTime);
    const loaded = loadTestResults();
    return loaded.accuracy === accuracy && loaded.loss === loss;
  }
), { numRuns: 100 });

// Feature: histosight-unified-dashboard, Property 9: Invalid accuracy rejected
fc.assert(fc.property(
  fc.oneof(
    fc.float({ max: -Number.EPSILON }),
    fc.float({ min: 1 + Number.EPSILON }),
    fc.constant(NaN),
    fc.string(),
  ),
  (invalidAcc) => {
    const before = localStorage.getItem('histosight_test_results');
    attemptSave(invalidAcc, 0.5, 100);
    return localStorage.getItem('histosight_test_results') === before;
  }
), { numRuns: 100 });
```

### Minimum Iterations

All property-based tests are configured to run a minimum of **100 iterations** per property.

### Integration / Smoke Tests

- Smoke: verify `MAX_TRAIN_SAMPLES <= 40000`, `TOTAL_EPOCHS <= 45`, `PATIENCE == 10` in `train.py`
- Smoke: verify `inference.py` is not modified (content hash check)
- Integration: start FastAPI with `TestClient`, call all five endpoints, verify response schemas
