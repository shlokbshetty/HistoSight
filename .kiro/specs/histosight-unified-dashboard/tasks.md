# Tasks

## Task List

- [x] 1. Fix nt_xent_loss in train.py
  - [x] 1.1 Add missing variable setup (z, N, sim, mask) before the existing masked_fill line in nt_xent_loss
  - [x] 1.2 Verify the labels tensor construction and F.cross_entropy call are correct

- [x] 2. Fix contrastive_top1_acc in train.py
  - [x] 2.1 Add missing variable setup (z, N, sim, mask) before the existing masked_fill line in contrastive_top1_acc
  - [x] 2.2 Verify the labels tensor and argmax accuracy computation are correct

- [x] 3. Reduce training hyperparameters in train.py
  - [x] 3.1 Change MAX_TRAIN_SAMPLES from 50000 to 40000
  - [x] 3.2 Change TOTAL_EPOCHS from 55 to 45

- [x] 4. Add POST /train_baseline endpoint to backend/main.py
  - [x] 4.1 Add module-level subprocess state variables (_baseline_proc, _baseline_done)
  - [x] 4.2 Implement POST /train_baseline with idle/running/done guard logic and subprocess.Popen spawn
  - [x] 4.3 Implement GET /train_baseline/status returning current status and resnet_supervised data

- [x] 5. Create unified index.html SPA
  - [x] 5.1 Write the HTML skeleton with Tailwind CDN, Chart.js CDN, Google Fonts, and Material Symbols imports
  - [x] 5.2 Implement the fixed sidebar with five nav links (Upload, Analytics, Comparison, Test, About) and data-section attributes
  - [x] 5.3 Implement the navigate() function that shows/hides sections and toggles active link styles
  - [x] 5.4 Implement the Upload section: drag-drop zone, file preview, POST /predict call, result display, error banner, disabled-button state
  - [x] 5.5 Implement the Analytics section: fetch GET /metrics on activation, render three Chart.js line charts (train loss, val loss, train accuracy), summary stat cards, no-data banner
  - [x] 5.6 Implement the Comparison section: fetch GET /comparison on activation, render Chart.js grouped bar chart, summary table, in-progress banner when all accuracies are 0
  - [x] 5.7 Implement the Test section: read-only card from localStorage, Edit/Save flow, validation for accuracy in [0,1], persist to histosight_test_results key
  - [x] 5.8 Implement the About section with static project metadata, hardware specs, and external links
  - [x] 5.9 Implement DOMContentLoaded handler: call POST /train_baseline once, log response, show toast on network error
  - [x] 5.10 Implement 30-second polling: setInterval fetching /metrics and /comparison concurrently, stop when /train_baseline/status returns "done", log errors without blocking UI
  - [x] 5.11 Implement responsive sidebar: collapse on viewports < 768px with hamburger toggle button

- [x] 6. Write property-based tests for nt_xent_loss and contrastive_top1_acc
  - [x] 6.1 Write Property 1 test: NT-Xent loss < log(2N-1) for perfect positive pairs (z1==z2), using Hypothesis with 100 examples
  - [x] 6.2 Write Property 2 test: contrastive_top1_acc returns float in [0.0, 1.0] for any valid z1/z2, using Hypothesis with 100 examples
  - [x] 6.3 Write Property 3 test: perfect pairs yield acc==1.0 and loss < log(2N-1) simultaneously, using Hypothesis with 100 examples

- [x] 7. Write property-based tests for frontend navigation and test section
  - [x] 7.1 Write Property 4 test: navigate(name) shows exactly one section, using fast-check with 100 runs
  - [x] 7.2 Write Property 5 test: navigate(name) applies active class to exactly one sidebar link, using fast-check with 100 runs
  - [x] 7.3 Write Property 8 test: localStorage round-trip for valid test results, using fast-check with 100 runs
  - [x] 7.4 Write Property 9 test: invalid accuracy values are rejected and localStorage is not updated, using fast-check with 100 runs

- [x] 8. Write example-based tests for backend endpoints
  - [x] 8.1 Write tests for POST /train_baseline: started (202), already_running (200), already_done (200)
  - [x] 8.2 Write tests for GET /train_baseline/status: idle, running, done states
  - [x] 8.3 Write regression tests verifying POST /predict, POST /batch_predict, GET /metrics, GET /comparison still respond correctly
