# Requirements Document

## Introduction

HistoSight Unified Dashboard is a browser-based single-page application (SPA) that provides a unified interface for the Stain-Invariant SimCLR + Macenko histopathology classification system trained on PatchCamelyon (PCam). The feature encompasses three interconnected concerns:

1. **Fix broken SimCLR training functions** — `nt_xent_loss` and `contrastive_top1_acc` in `train.py` have incomplete bodies (missing variable setup for `z`, `N`, `sim`). These must be corrected so SSL pre-training runs end-to-end.
2. **Reduce SSL pre-training wall-clock time** — Current configuration (~5 h on RTX 4050 6 GB) must be tuned to fit within ~4.5 h by adjusting `MAX_TRAIN_SAMPLES`, `TOTAL_EPOCHS`, and related hyperparameters without degrading convergence quality.
3. **Unify the frontend into a single SPA** — The four separate HTML page fragments in `stitch_frontend_prompt.md` (Upload, Analytics, Comparison, Test) must be merged into one `index.html` file using vanilla JS, Tailwind CDN, and Chart.js, with sidebar navigation that switches sections without a page reload.
4. **Auto-start supervised baseline** — When the SPA loads, it must automatically trigger `train_supervised_baseline.py` via a new `POST /train_baseline` backend endpoint so the Model Comparison chart receives real data without manual CLI intervention.
5. **Live training status polling** — The frontend must periodically refresh chart data from `/metrics` and `/comparison` while training is in progress.

---

## Glossary

- **SPA**: Single-Page Application — one HTML file whose visible content changes via JavaScript without full page reloads.
- **SimCLR**: Simple Contrastive Learning of Representations — the self-supervised pre-training algorithm implemented in `train.py`.
- **NT-Xent Loss**: Normalized Temperature-scaled Cross-Entropy loss — the contrastive objective used by SimCLR.
- **Macenko**: Stain normalization algorithm applied before SimCLR augmentations to reduce stain-color variance.
- **PCam**: PatchCamelyon — the histopathology patch dataset used for training and evaluation.
- **Encoder**: The ResNet-18 backbone that produces 512-dimensional feature vectors.
- **Projector**: The two-layer MLP head that maps encoder features to the 128-dimensional contrastive embedding space.
- **Projection Vector (`z`)**: The L2-normalized output of the Projector for a single augmented view.
- **Positive Pair**: The two augmented views (`z1`, `z2`) of the same image, which the NT-Xent loss pulls together.
- **Negative Pair**: Any two projection vectors from different images within the same batch.
- **Temperature (`τ`)**: Scalar that sharpens or softens the similarity distribution in NT-Xent loss (default 0.5).
- **Top-1 Contrastive Accuracy**: Fraction of samples for which the positive pair ranks first among all negatives.
- **Baseline**: The fully supervised ResNet-18 trained in `train_supervised_baseline.py`, used as a comparison reference.
- **Dashboard**: The browser-based SPA (`index.html`) served alongside the FastAPI backend.
- **Backend**: The FastAPI application in `backend/main.py`.
- **Polling Interval**: The fixed time interval at which the SPA re-fetches `/metrics` and `/comparison` (30 seconds).
- **Section**: One of the five named views within the SPA — Upload, Analytics, Comparison, Test, About.
- **Active Section**: The Section currently visible in the main content area.
- **Sidebar**: The fixed left-hand navigation panel listing all Sections.
- **Chart.js**: The CDN-loaded JavaScript charting library used to render line and bar charts.
- **Tailwind CDN**: The CDN-loaded Tailwind CSS utility framework used for styling.
- **`/train_baseline` Endpoint**: The new `POST` route added to the Backend that spawns `train_supervised_baseline.py` as a background subprocess.
- **Training Status**: A tri-state value — `idle`, `running`, or `done` — reported by the Backend for the baseline training job.

---

## Requirements

### Requirement 1: Fix NT-Xent Loss Function

**User Story:** As an ML engineer, I want the `nt_xent_loss` function in `train.py` to be fully implemented, so that SimCLR pre-training can run without runtime errors.

#### Acceptance Criteria

1. THE `nt_xent_loss` function SHALL accept two L2-normalized projection tensors `z1` and `z2`, each of shape `(N, D)`, and a scalar `temperature` parameter.
2. WHEN `nt_xent_loss` is called, THE function SHALL concatenate `z1` and `z2` along dimension 0 to form a combined tensor `z` of shape `(2N, D)`.
3. WHEN `nt_xent_loss` is called, THE function SHALL compute the `(2N, 2N)` cosine similarity matrix `sim` by dividing the dot-product matrix `z @ z.T` by `temperature`.
4. WHEN `nt_xent_loss` is called, THE function SHALL construct a boolean self-similarity mask of shape `(2N, 2N)` and apply it to `sim` by filling masked positions with `-1e4` before computing the cross-entropy loss.
5. WHEN `nt_xent_loss` is called, THE function SHALL assign positive-pair labels such that row `i` (for `i < N`) targets column `i + N`, and row `i + N` targets column `i`.
6. WHEN `nt_xent_loss` is called with a batch of `N` identical image pairs (z1 == z2), THE function SHALL return a loss value strictly less than `log(2N - 1)` (the maximum possible loss), confirming that positive pairs are ranked above at least some negatives.
7. IF `z1` or `z2` contains NaN or Inf values, THEN THE `nt_xent_loss` function SHALL propagate the invalid values naturally (no silent masking), allowing the training loop's gradient clipping to handle instability.

---

### Requirement 2: Fix Contrastive Top-1 Accuracy Function

**User Story:** As an ML engineer, I want the `contrastive_top1_acc` function in `train.py` to be fully implemented, so that training logs report a meaningful accuracy metric each epoch.

#### Acceptance Criteria

1. THE `contrastive_top1_acc` function SHALL accept two detached L2-normalized projection tensors `z1` and `z2`, each of shape `(N, D)`, and a scalar `temperature` parameter.
2. WHEN `contrastive_top1_acc` is called, THE function SHALL reuse the same `z`, `N`, `sim`, and `mask` construction logic as `nt_xent_loss` to produce a masked similarity matrix.
3. WHEN `contrastive_top1_acc` is called, THE function SHALL compute `labels` using the same positive-pair indexing as `nt_xent_loss`.
4. WHEN `contrastive_top1_acc` is called, THE function SHALL return the fraction of rows in `sim` for which `argmax` equals the corresponding label, as a Python `float` in the range `[0.0, 1.0]`.
5. WHEN both functions are called on the same inputs, THE `contrastive_top1_acc` function SHALL produce an accuracy value that is consistent with the loss value from `nt_xent_loss` (i.e., lower loss correlates with higher accuracy across training epochs).

---

### Requirement 3: Reduce SSL Pre-Training Time to ≤ 4.5 Hours

**User Story:** As an ML engineer, I want the SimCLR pre-training to complete within approximately 4.5 hours on the RTX 4050 6 GB GPU, so that the full pipeline (pre-train → fine-tune → evaluate) fits within a single working session.

#### Acceptance Criteria

1. THE `train.py` configuration SHALL set `MAX_TRAIN_SAMPLES` to a value no greater than 40000 (reduced from 50000).
2. THE `train.py` configuration SHALL set `TOTAL_EPOCHS` to a value no greater than 45 (reduced from 55).
3. WHEN training runs on the RTX 4050 6 GB GPU with the updated configuration, THE training loop SHALL complete all epochs within 16200 seconds (4.5 hours) of wall-clock time.
4. THE updated configuration SHALL preserve the existing early-stopping mechanism (`PATIENCE = 10`) so training can terminate before the epoch limit if validation loss plateaus.
5. THE updated configuration SHALL preserve the existing batch-size fallback mechanism (64 → 32) to handle GPU OOM conditions.
6. WHERE the `--epochs` CLI argument is provided, THE `train.py` script SHALL use the CLI value instead of the constant, allowing manual override without code changes.

---

### Requirement 4: Unified Single-Page Application

**User Story:** As a clinician or researcher, I want a single HTML file that contains all dashboard sections, so that I can navigate between Upload, Analytics, Comparison, Test, and About without page reloads or a build step.

#### Acceptance Criteria

1. THE Dashboard SHALL be delivered as a single `index.html` file that requires no build step and loads all dependencies (Tailwind CDN, Chart.js CDN, Google Fonts, Material Symbols) from CDN URLs.
2. THE Dashboard SHALL render a fixed Sidebar on the left containing navigation links for the five Sections: Upload, Analytics, Comparison, Test, and About.
3. WHEN a Sidebar navigation link is clicked, THE Dashboard SHALL show the corresponding Section's content in the main area and hide all other Sections' content without a page reload.
4. WHEN a Sidebar navigation link is clicked, THE Dashboard SHALL update the Active Section's link to the active visual style (teal highlight, filled icon) and remove the active style from all other links.
5. THE Dashboard SHALL apply the dark-mode color palette defined in `stitch_frontend_prompt.md` (background `#111319`, cards `#1e1f26`, accent `#45e4e4`) consistently across all Sections.
6. THE Dashboard SHALL be responsive such that the main content area occupies the full viewport width minus the 256 px Sidebar width on screens wider than 768 px.
7. IF the browser viewport is narrower than 768 px, THEN THE Dashboard SHALL collapse the Sidebar and provide a hamburger toggle to show or hide it.

---

### Requirement 5: Upload & Predict Section

**User Story:** As a clinician, I want to drag-and-drop or browse for a histopathology image and receive an immediate tumor/normal prediction with confidence score, so that I can triage slides efficiently.

#### Acceptance Criteria

1. THE Upload Section SHALL provide a drag-and-drop zone that accepts JPEG, PNG, and TIFF image files.
2. WHEN a file is dropped or selected via the browse button, THE Upload Section SHALL display a thumbnail preview of the selected image.
3. WHEN the "Run Prediction" button is clicked, THE Upload Section SHALL POST the image to `POST /predict` and display a loading spinner until the response is received.
4. WHEN a prediction response is received, THE Upload Section SHALL display the predicted class ("Tumor" or "Normal") and the confidence percentage.
5. IF the `/predict` endpoint returns an HTTP error, THEN THE Upload Section SHALL display an inline error banner with the error message.
6. IF no file has been selected, THEN THE Upload Section SHALL disable the "Run Prediction" button and display a placeholder state in the result area.

---

### Requirement 6: Training Analytics Section

**User Story:** As an ML engineer, I want to view line charts of training loss, validation loss, and accuracy over epochs, so that I can monitor SimCLR pre-training convergence.

#### Acceptance Criteria

1. WHEN the Analytics Section becomes the Active Section, THE Dashboard SHALL fetch data from `GET /metrics` and render three Chart.js line charts: Train Loss vs. Epoch, Val Loss vs. Epoch, and Train Accuracy vs. Epoch.
2. WHEN the `/metrics` fetch succeeds and the response contains at least one epoch, THE Analytics Section SHALL render the charts with the epoch array on the x-axis.
3. IF the `/metrics` endpoint returns a 404 or network error, THEN THE Analytics Section SHALL display a "No training data available" banner instead of the charts.
4. WHILE the Polling Interval timer is active, THE Dashboard SHALL re-fetch `/metrics` every 30 seconds and update the chart data in place without a full page reload.
5. THE Analytics Section SHALL display the most recent epoch number, most recent train loss, and most recent val loss as summary stat cards above the charts.

---

### Requirement 7: Model Comparison Section

**User Story:** As an ML engineer, I want to see a bar chart comparing the accuracy of the supervised ResNet-18 baseline, the SimCLR model, and the final fine-tuned model, so that I can quantify the benefit of self-supervised pre-training.

#### Acceptance Criteria

1. WHEN the Comparison Section becomes the Active Section, THE Dashboard SHALL fetch data from `GET /comparison` and render a Chart.js grouped bar chart with three bars: ResNet Supervised, SimCLR (no Macenko), and Final Model.
2. WHEN the `/comparison` fetch succeeds, THE Comparison Section SHALL display each model's accuracy as a percentage on the bar chart y-axis.
3. IF all three accuracy values in the comparison response are 0.0, THEN THE Comparison Section SHALL display an "In Progress — baseline training running" status banner alongside the (empty) chart.
4. WHILE the Polling Interval timer is active, THE Dashboard SHALL re-fetch `/comparison` every 30 seconds and update the bar chart data in place.
5. THE Comparison Section SHALL display a summary table below the chart listing each model's accuracy, training time (seconds), and epoch count.

---

### Requirement 8: Auto-Trigger Supervised Baseline Training

**User Story:** As an ML engineer, I want the supervised baseline training to start automatically when the dashboard loads, so that the Model Comparison chart populates with real data without requiring a manual CLI command.

#### Acceptance Criteria

1. THE Backend SHALL expose a `POST /train_baseline` endpoint that spawns `train_supervised_baseline.py` as a non-blocking background subprocess.
2. WHEN `POST /train_baseline` is called and no baseline training subprocess is currently running, THE Backend SHALL start the subprocess and return `{"status": "started"}` with HTTP 202.
3. WHEN `POST /train_baseline` is called and a baseline training subprocess is already running, THE Backend SHALL return `{"status": "already_running"}` with HTTP 200 without spawning a second process.
4. WHEN `POST /train_baseline` is called and the baseline training has already completed (i.e., `comparison.json` contains a non-zero `resnet_supervised.accuracy`), THE Backend SHALL return `{"status": "already_done"}` with HTTP 200 without spawning a new process.
5. THE Backend SHALL expose a `GET /train_baseline/status` endpoint that returns the current Training Status (`idle`, `running`, or `done`) and the `resnet_supervised` entry from `comparison.json`.
6. WHEN the Dashboard loads (DOMContentLoaded), THE Dashboard SHALL call `POST /train_baseline` once and log the response status to the browser console.
7. IF the `POST /train_baseline` call fails with a network error, THEN THE Dashboard SHALL silently continue loading without blocking the UI, and display a non-intrusive warning toast.

---

### Requirement 9: Training Status Polling

**User Story:** As an ML engineer, I want the dashboard charts to refresh automatically while training is running, so that I can see live progress without manually reloading the page.

#### Acceptance Criteria

1. WHEN the Dashboard loads, THE Dashboard SHALL start a Polling Interval timer that fires every 30 seconds.
2. WHEN the Polling Interval fires, THE Dashboard SHALL fetch `/metrics` and `/comparison` concurrently and update all rendered charts with the latest data.
3. WHEN the Polling Interval fires and the Active Section is Analytics, THE Dashboard SHALL update the Analytics charts and summary stat cards with the new data.
4. WHEN the Polling Interval fires and the Active Section is Comparison, THE Dashboard SHALL update the Comparison bar chart and summary table with the new data.
5. WHEN `GET /train_baseline/status` returns `{"status": "done"}`, THE Dashboard SHALL stop the Polling Interval timer.
6. IF a polling fetch fails with a network error, THEN THE Dashboard SHALL log the error to the browser console and retry on the next interval without displaying a blocking error to the user.

---

### Requirement 10: Test Results Section

**User Story:** As an ML engineer, I want a read-only card that displays the final test-phase evaluation results, so that I can record and share the model's held-out performance.

#### Acceptance Criteria

1. THE Test Section SHALL display a read-only card showing test accuracy, test loss, and inference time, populated from `localStorage` if previously saved.
2. WHEN the Test Section is first loaded and `localStorage` contains no saved test results, THE Test Section SHALL display placeholder dashes for each metric.
3. THE Test Section SHALL provide an "Edit" button that reveals input fields for manually entering test accuracy, test loss, and inference time.
4. WHEN the "Save" button is clicked after editing, THE Test Section SHALL persist the entered values to `localStorage` under the key `histosight_test_results` and return to the read-only card view.
5. IF the entered test accuracy value is not a number in the range `[0, 1]`, THEN THE Test Section SHALL display an inline validation error and prevent saving.

---

### Requirement 11: About Section

**User Story:** As a project stakeholder, I want an About section that documents the project metadata and hardware configuration, so that results can be reproduced and attributed correctly.

#### Acceptance Criteria

1. THE About Section SHALL display the project name ("HistoSight"), model architecture ("ResNet-18 + SimCLR + Macenko"), dataset ("PatchCamelyon (PCam)"), and framework versions (PyTorch, torchstain, FastAPI).
2. THE About Section SHALL display the hardware specification: GPU model ("RTX 4050 Laptop GPU, 6 GB VRAM"), CPU model ("AMD Ryzen 7 7735HS"), and training environment ("Python 3.12, CUDA").
3. THE About Section SHALL display a link to the PCam dataset reference and a link to the SimCLR paper.

---

### Requirement 12: Preserve All Existing Backend Functionality

**User Story:** As a developer, I want all existing API endpoints to continue working after the new `/train_baseline` endpoint is added, so that no regression is introduced.

#### Acceptance Criteria

1. THE Backend SHALL continue to serve `POST /predict`, `POST /batch_predict`, `GET /metrics`, and `GET /comparison` with the same request/response contracts as before the changes.
2. THE Backend SHALL continue to import and use `inference.py` for prediction endpoints without modification to `inference.py`.
3. WHEN the Backend starts, THE Backend SHALL not automatically spawn any training subprocess; subprocess spawning SHALL only occur in response to an explicit `POST /train_baseline` request.
