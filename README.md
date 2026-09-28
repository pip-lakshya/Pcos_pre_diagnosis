# PCOS screening companion — Phase 2

A conversational intake assistant for a **screening estimate**, with a model-backed explanation. It is not a diagnostic tool. Chat and browser speech input/output are included. The system prompt and visible UI both explain that the result is not a diagnosis and recommend consulting a doctor.

## Single source of truth

All project assets now live under `pcos-agent/`. The balanced Random Forest pickle already loaded by the backend was retained as the canonical model. Its path remains explicit in `backend/app/config.py` as `backend/ml/pcos_rf_model.pkl`; the SMOTENC model is isolated in `backend/ml/experiments/` and is not referenced by the app.

```text
pcos-agent/
├── README.md
├── TODO.md
├── backend/
│   ├── app/                         # FastAPI, chat agent, model predictor
│   ├── .env.example
│   ├── requirements.txt
│   ├── requirements-notebooks.txt   # additional notebook tooling
│   └── ml/
│       ├── data/
│       │   ├── PCOS_data_without_infertility.xlsx
│       │   └── PCOS_infertility.csv
│       ├── notebooks/
│       │   ├── pcos_random_forest_og_rf.ipynb
│       │   └── pcos_random_forest_smotenc.ipynb
│       ├── experiments/
│       │   └── pcos_rf_smotenc_model.pkl
│       ├── pcos_rf_model.pkl         # canonical; class_weight="balanced"
│       ├── pcos_model_features.json  # canonical feature order
│       └── train_pcos_model.py
└── frontend/                        # React + Vite + TypeScript + Tailwind
```

The former root-level copies were removed after verifying the new copies and running both notebooks against the relocated workbook. The old generic duplicate notebook in `backend/ml/` was also removed in favor of the two explicitly named source notebooks.

## Run locally

Requires Python 3.10+ with pip and Node.js/npm. Package downloads must be available for first-time setup.

```bash
cd pcos-agent/backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your NVIDIA_API_KEY
uvicorn app.main:app --reload
```

In a second terminal:

```bash
cd pcos-agent/frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Set `VITE_API_URL` in `frontend/.env` if the API is not at `http://localhost:8000`. Browser speech recognition availability depends on the browser and its speech service; typed chat remains available. `VoiceProvider` in `frontend/src/hooks/useVoice.ts` isolates the UI from the browser implementation.

## Configuration

- `NVIDIA_API_KEY`: required for natural-language feature extraction and question generation.
- `NVIDIA_MODEL`: NIM model name; defaults to `openai/gpt-oss-20b`.
- `NVIDIA_BASE_URL`: defaults to `https://integrate.api.nvidia.com/v1`.
- `CORS_ORIGINS`: comma-separated frontend origins; defaults to `http://localhost:5173`.
- `JWT_SECRET`: required random signing secret for seven-day bearer tokens. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- `DATABASE_URL`: SQLite URL by default; the MVP creates `backend/pcos_agent.sqlite3` automatically.

Keep `.env` local and do not commit the key. Without an API key, the chat can produce deterministic question wording but cannot extract answers; configure NIM for a complete conversational intake.

## Data, notebooks, and training

Both notebooks read the workbook from `../data/PCOS_data_without_infertility.xlsx` and save model artifacts into their appropriate `ml/` locations. Execute them from `backend/ml/notebooks/` so those relative paths resolve. Install notebook dependencies with `pip install -r ../requirements-notebooks.txt` from that directory, then run:

```bash
jupyter nbconvert --to notebook --execute --output-dir /tmp/pcos-notebook-results pcos_random_forest_og_rf.ipynb
jupyter nbconvert --to notebook --execute --output-dir /tmp/pcos-notebook-results pcos_random_forest_smotenc.ipynb
```

The balanced notebook writes the canonical model and feature list. The SMOTENC notebook writes its model to `ml/experiments/` and writes the shared feature list. `backend/ml/train_pcos_model.py` trains the balanced model from the bundled workbook and intentionally replaces the canonical model and feature JSON; it can be run from any working directory with `python backend/ml/train_pcos_model.py` from `pcos-agent/`.

## Model inputs and output

The extractor forces the `extract_pcos_features` tool on every `/chat` turn. Its nullable/optional schema contains only 14 raw fields: `age`, `weight_kg`, `height_cm`, `cycle_length_days`, `weight_gain`, `hair_growth`, `skin_darkening`, `hair_loss`, `acne`, `fast_food`, `regular_exercise`, `hip_inch`, `waist_inch`, and `cycle_irregular`. Boolean answers are extracted as booleans and normalized by Python to 0/1. BMI and waist:hip ratio are excluded from the tool schema; Python derives `bmi = weight_kg / (height_cm / 100)^2` and `waist_hip_ratio = waist_inch / hip_inch` when both inputs are present.

The unchanged feature JSON order is the model contract:

1. `age`
2. `weight_kg`
3. `height_cm`
4. `cycle_length_days`
5. `weight_gain`
6. `hair_growth`
7. `skin_darkening`
8. `hair_loss`
9. `acne`
10. `fast_food`
11. `regular_exercise`
12. `hip_inch`
13. `waist_inch`
14. `cycle_irregular`
15. `bmi`
16. `waist_hip_ratio`

Hip and waist measures are optional in chat. The trained model requires every input column, so an explicitly skipped/unavailable optional circumference and its ratio are represented as `0.0` at inference. This MVP assumption may affect the estimate. `/predict` expects `{ "features": { ...all 16 keys... } }` and returns probability, `risk_label` (`elevated` at probability >= 0.5, otherwise `lower`), and feature importances. The probability is a model score, not a calibrated clinical risk unless separately validated.

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"age":25,"weight_kg":62,"height_cm":165,"cycle_length_days":35,"weight_gain":0,"hair_growth":0,"skin_darkening":0,"hair_loss":0,"acne":1,"fast_food":0,"regular_exercise":1,"hip_inch":38,"waist_inch":31,"cycle_irregular":1,"bmi":22.77,"waist_hip_ratio":0.816}}'
```

For `/chat`, send `{"message":"I am 25 years old"}` to `http://localhost:8000/chat`, then pass the returned `session_id` in each following request.

## Accounts, screening history, and research questions

Register with `POST /auth/register` or sign in with `POST /auth/login` using `{ "email": "...", "password": "..." }`. Both return a JWT as `access_token`. Passwords are stored as bcrypt hashes. The frontend keeps the token in localStorage, checks it with `GET /auth/me` on load, and sends `Authorization: Bearer <token>` on protected calls. Keep the local JWT secret private and set a new secret in production.

Authenticated `/chat` conversations keep their live slot state in memory scoped by both user ID and session ID. When all model inputs are available and the Random Forest completes, the result is persisted to the `screenings` table. `GET /screenings/me` returns the current user's completed results, newest first. Since live slot state is intentionally in memory for this MVP, an unfinished conversation does not survive a backend restart; completed screening history does.

After a screening, `POST /chat/research` accepts `{ "question": "..." }`. It retrieves relevant chunks from the curated markdown under `backend/app/agent/knowledge/`, then asks the configured NVIDIA model to answer using only those chunks. If retrieval finds no relevant material, the service declines rather than asking the model to answer from general knowledge. This provider is selected through a `ResearchProvider` interface; `KnowledgeBaseProvider` is the local implementation. It reads the latest saved screening only to personalize context and never calls, changes, or blends with the Random Forest prediction. Research replies include source references and varied screening-not-diagnosis / consult-a-doctor language. The documents cover overview, symptoms, possible contributors, lifestyle, when to seek care, and common misconceptions, with sources linked in each document.

Example authenticated request:

```bash
curl -X POST http://localhost:8000/chat/research \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"What foods should I avoid?"}'
```

## Verification performed

- Audited source copies before changing anything. The original canonical pickle and feature JSON were byte-identical to the copies previously used by the backend; the SMOTENC artifact is a distinct pipeline.
- Ran all 12 non-empty code cells in the balanced notebook and all 13 in the SMOTENC notebook headlessly from the new `ml/notebooks/` directory. All cells passed. `nbconvert` is not installed in the available environment; execution used the installed IPython kernel directly.
- After restoring the original canonical pickle, loaded `backend/ml/pcos_rf_model.pkl` from its final location with the existing Python environment. Confirmed it is a `RandomForestClassifier` with `class_weight="balanced"`, 16 inputs, and that a complete sample row returns valid two-class `predict_proba` values.
- `python3 -m compileall` passed for the backend app and the new training script.
- The app imports and the modified modules pass `compileall`. The direct NVIDIA forced-tool request timed out at both 45 and 120 seconds without a response, so the live NIM extraction integration could not be confirmed in this environment.
- A three-turn in-process `/chat` handler run with a mock NIM verified sparse non-null extraction updates, follow-up questions for missing slots, the canonical model prediction, and a natural-language final explanation. Assertions confirmed each extraction call forced `extract_pcos_features`, while follow-up and final explanation calls supplied neither `tools` nor `tool_choice`.
- `npm run build` passes after the auth, history, and research UI changes.
- Authenticated API integration was exercised with SQLite and the real canonical Random Forest; account creation/login, wrong-password rejection, 401 protection, multi-turn state, derived values, automatic model prediction, persisted history, relevant research retrieval, and out-of-scope refusal passed. The NIM extractor and prose generation were mocked for this integration test because direct calls to NVIDIA timed out in this environment; do not treat the mock run as live NIM verification.
