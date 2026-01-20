# Melanoma Detection – Core

This repository bundles the AI services and the Expo/React Native client used to run a melanoma screening assistant. The project trains an EfficientNet-B0 classifier on HAM10000 dermoscopic images, exposes the model through a FastAPI backend, and delivers the result to a mobile UI that lets users photograph a mole and receive a risk estimate (with a clear medical disclaimer).

## Repository layout

- `melanoma-detection-ai/` – Python 3.11 project with notebooks, training scripts (`ml/`), serialized artifacts (`artifacts/efficientnet_b0_best.pt`) and a FastAPI service (`api/`) exposing `/health` and `/predict`.
- `melanoma-detection-app/` – Expo/React Native client (Node 20 + pnpm) that lets you pick or take a photo, uploads it to the backend and displays the probability plus warning copy.
- `sprawozdanie.md` – project report in Polish that documents the methodology and evaluation.

Each sub-project ships its own README with deeper configuration notes; start here for a birds-eye view and quick start.

## Key decisions

- **Dataset** – HAM10000 downsampled to a binary task (`mel` vs. benign). The train/val/test split is stratified 80/10/10 to avoid leakage.
- **Model** – EfficientNet-B0 with ImageNet weights, a sigmoid head, BCEWithLogitsLoss + `pos_weight`, AdamW and ReduceLROnPlateau. Early stopping arrests overfitting.
- **Serving** – FastAPI handles preprocessing, model inference (CPU/GPU), and returns both the probability and a `low_risk`/`high_risk` label; it also repeats the “not a diagnosis” disclaimer.
- **Client UX** – Expo app rewrites localhost to `10.0.2.2` for Android emulators, reads the backend URL from `EXPO_PUBLIC_API_URL`, and shows clear toasts for validation errors.

## Quick start

1. **Backend**
   ```bash
   cd melanoma-detection-ai
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Verify with `curl http://127.0.0.1:8000/health`. The model loads from `artifacts/efficientnet_b0_best.pt`. For one-command setup see `scripts/dev.sh`.

2. **Mobile app**
   ```bash
   cd melanoma-detection-app
   cp .env.example .env   # set EXPO_PUBLIC_API_URL, e.g. http://192.168.0.42:8000
   pnpm install
   pnpm start
   ```
   Press `i`/`a` to launch an iOS/Android simulator or scan the QR code with Expo Go. On Android emulators keep the backend URL as `http://localhost:8000`; the app rewrites it as needed.

3. **Upload & result**
   - In Expo, open the **Upload** tab, choose a mole photo or take one with the camera.
   - Tap **Analyze**; the app posts a `multipart/form-data` request to `POST /predict`.
   - The backend responds with a probability and label, which the app shows on the **Result** screen.

## Development notes

- Training code, evaluation metrics, and experiments with “phone-like” augmentations live under `melanoma-detection-ai/ml/`; see that README for thresholds and experiment logs.
- The FastAPI service expects the artifact path to remain stable; update `api/config.py` if you relocate checkpoints.
- The Expo app relies on pnpm workspaces and Expo Router. TypeScript sources live in `app/` & `components/`; shared helpers are under `lib/`.

For troubleshooting, consult the sub-project READMEs—they contain platform-specific fixes, environment tips, and more exhaustive setup guides.

