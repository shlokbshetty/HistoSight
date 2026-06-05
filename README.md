# HistoSight — Stain-Invariant SSL for Histopathology Classification

Binary tumor/normal classification on PatchCamelyon (PCam) using SimCLR self-supervised pre-training + linear evaluation.

---

## Project Structure

```
project/
├── index.html                  # Unified dashboard SPA (open in browser)
├── train.py                    # SimCLR SSL pre-training
├── fine_tune.py                # Linear evaluation on frozen encoder
├── evaluate.py                 # Test-phase evaluation (accuracy + loss)
├── inference.py                # Single/batch prediction (used by backend)
├── train_supervised_baseline.py # ResNet18 supervised baseline
├── backend/
│   └── main.py                 # FastAPI server
├── simclr/ssl_splits/          # PCam HDF5 dataset files
├── checkpoints/                # Saved model checkpoints
├── models/                     # Saved classifier weights
├── logs/
│   ├── metrics.json            # Per-epoch SSL training metrics
│   └── comparison.json         # Model comparison data
└── tests/                      # Property-based + example tests
```

---

## Setup

```bash
# Install dependencies
venv312\Scripts\pip install -r requirements.txt

# Optional: Macenko stain normalization
venv312\Scripts\pip install torchstain
```

---

## How to Run Everything

### Step 1 — SSL Pre-Training (SimCLR)

```bash
venv312\Scripts\python.exe train.py
```

- Loads 20k samples into RAM, trains 30 epochs (~45-60 min on RTX 4050)
- Saves checkpoints to `checkpoints/` every epoch
- Best checkpoint saved to `checkpoints/best_simclr.pth`
- Updates `logs/metrics.json` after every epoch
- Evaluates pretrained ResNet18 baseline once at startup → `logs/comparison.json`

**If accuracy is low after 30 epochs, resume training for more epochs:**

```bash
# Resume from checkpoint and train up to 50 epochs total
venv312\Scripts\python.exe train.py --epochs 50
```

The script automatically resumes from `checkpoints/best_simclr.pth` — it will NOT restart from scratch.

**To run even more epochs:**

```bash
venv312\Scripts\python.exe train.py --epochs 80
```

### Step 2 — Fine-Tuning (Linear Evaluation)

After SSL pre-training completes, train a linear classifier on top of the frozen encoder:

```bash
venv312\Scripts\python.exe fine_tune.py
```

- Loads frozen encoder from `checkpoints/best_simclr.pth`
- Trains `nn.Linear(512, 2)` on labeled val split
- Saves best classifier to `models/best_classifier.pth`
- Updates `logs/comparison.json` → `final_model` entry

### Step 3 — Evaluate on Test Set

```bash
venv312\Scripts\python.exe evaluate.py
```

Outputs only:
```
Test Accuracy : XX.XX%
Test Loss     : X.XXXX
```

### Step 4 — Start the Backend

```bash
venv312\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### Step 5 — Open the Dashboard

Open `index.html` directly in your browser (double-click or drag into Chrome/Edge).

The dashboard will:
- Auto-trigger supervised baseline training via `POST /train_baseline`
- Poll `/metrics` and `/comparison` every 30 seconds
- Show all charts from stored JSON data

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload image → class + confidence |
| `POST` | `/batch_predict` | Folder path → predictions + accuracy |
| `GET`  | `/metrics` | Returns `logs/metrics.json` |
| `GET`  | `/comparison` | Returns `logs/comparison.json` |
| `POST` | `/train_baseline` | Start supervised baseline training |
| `GET`  | `/train_baseline/status` | Check baseline training status |

---

## What the Accuracy Numbers Mean

The SSL pre-training reports **contrastive top-1 accuracy** — this is NOT classification accuracy. It measures how often the model correctly identifies the positive pair (same image, different augmentation) among all negatives in the batch.

- **~14% contrastive acc after 30 epochs** — normal for SimCLR with batch 256 on 20k samples
- **Real classification accuracy** comes from `fine_tune.py` (Step 2)
- **Pretrained ResNet18 baseline: ~85%** — this is the target to beat with SSL

### If classification accuracy after fine-tuning is low (< 70%):

1. Train more SSL epochs first:
   ```bash
   venv312\Scripts\python.exe train.py --epochs 60
   ```

2. Then re-run fine-tuning:
   ```bash
   venv312\Scripts\python.exe fine_tune.py
   ```

3. Repeat until fine-tuned accuracy improves. Each additional 30 epochs of SSL typically adds 2-5% classification accuracy.

---

## Running Tests

### Python property-based tests (requires `hypothesis`):

```bash
venv312\Scripts\pip install hypothesis pytest
venv312\Scripts\python.exe -m pytest tests/test_train_properties.py -v
```

### Backend endpoint tests:

```bash
venv312\Scripts\pip install httpx
venv312\Scripts\python.exe -m pytest tests/test_backend_endpoints.py -v
```

### Frontend property tests (requires Node.js):

```bash
npm install --save-dev jest jest-environment-jsdom fast-check
npx jest tests/test_frontend_properties.js
```

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| CPU | AMD Ryzen 7 7735HS |
| RAM | 16 GB+ |
| OS | Windows, Python 3.12, CUDA |

---

## Training Time Reference

| Phase | Time |
|-------|------|
| SSL pre-training (30 epochs, 20k samples) | ~45-60 min |
| SSL pre-training (60 epochs, 20k samples) | ~90-120 min |
| Fine-tuning (30 epochs) | ~10-15 min |
| Supervised baseline | ~15-20 min |
| Pretrained ResNet18 eval | ~1 min |

## Acknowledgments

**Research Team:** Shashank Tandan, Shlok Shetty, Shree Santosh Yadav, Siddharth S Gadekar

**Institution:** School of Computer Science and Engineering, RV University, Bengaluru

---

## Contact

For questions or support, please open an issue on the repository or contact the development team.