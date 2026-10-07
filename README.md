# Narisaarthi — Empower Your PCOS Journey

A conversational intake assistant for a **screening estimate**, with a model-backed explanation. It is not a diagnostic tool. Chat and browser speech input/output are included. The system prompt and visible UI both explain that the result is not a diagnosis and recommend consulting a doctor.

## Single source of truth

All project assets now live under `pcos-agent/`. The balanced Random Forest pickle remains the canonical training baseline. Inference uses a separate five-member seeded Random Forest ensemble saved under `backend/ml/ensemble/`; the members use the same approved 16 features and training split with distinct random seeds. The SMOTENC model is isolated in `backend/ml/experiments/` and is not referenced by the app.

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
│       │   ├── pcos_random_forest_smotenc.ipynb
│       │   └── 11_Engineered_Features_Comparison.ipynb
│       ├── ensemble/               # five distinct seeded inference models
│       ├── experiments/
│       │   └── pcos_rf_smotenc_model.pkl
│       ├── pcos_rf_model.pkl         # canonical; class_weight="balanced"
│       ├── pcos_model_features.json  # canonical feature order
│       ├── train_pcos_model.py      # unchanged canonical baseline trainer
│       └── train_ensemble.py        # builds inference ensemble separately
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

Open the Vite URL (normally `http://localhost:5173`). The frontend defaults to `http://127.0.0.1:8000` to match the backend's default Uvicorn bind; set `VITE_API_URL` in `frontend/.env` if your backend uses another host or port. Browser speech recognition availability depends on the browser and its speech service; typed chat remains available. `VoiceProvider` in `frontend/src/hooks/useVoice.ts` isolates the UI from the browser implementation.

## Production deployment

The deployable application is the existing `backend/` FastAPI service plus the static Vite site built from `frontend/`. Do not deploy `frontend/node_modules/`, `frontend/dist/` from a developer machine, local `.env` files, Python caches, or the local SQLite database. The root `.gitignore` excludes these generated and machine-local files while retaining source, tests, model artifacts, training assets, and notebooks.

1. Create the backend service with its working directory set to `backend/`. Install `backend/requirements.txt` and start it with `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (use the host's equivalent port variable if needed).
2. Add backend environment variables from `backend/.env.example` in the hosting provider's secret/configuration store. Set a unique high-entropy `JWT_SECRET`, the real `NVIDIA_API_KEY`, and `CORS_ORIGINS` to the exact deployed website origin(s). For Render Free, configure `EMAIL_PROVIDER=resend`, `RESEND_API_KEY`, `EMAIL_FROM` using a verified sender domain, and `ADMIN_NOTIFY_EMAIL`; Gmail SMTP is only for local development or a host that permits outbound SMTP.
3. Attach persistent storage for SQLite and set `DATABASE_URL` to that mounted location, for example `sqlite:////var/lib/narisaarthi/pcos_agent.sqlite3`. Ensure the directory exists and is writable before starting the service. The development database at `backend/pcos_agent.sqlite3` is local user data and must not be copied into a public image or committed.
4. Build the frontend from `frontend/` with `npm ci && npm run build`, setting `VITE_API_URL` to the public HTTPS backend origin at build time. Publish `frontend/dist/` with a static host configured to serve `index.html` for app routes such as `/app`, `/login`, `/register`, `/privacy`, and `/legal`.
5. Use HTTPS on both origins. Check `GET /health`, account registration/login, a screening, history, contact mail configuration, and PWA installation on the deployed domains. The browser's microphone/geolocation prompts and PWA install behavior require a secure context; NVIDIA and optional SMTP/TTS services also need outbound network access.

The runtime ensemble is loaded from `backend/ml/ensemble/`, while the baseline pickle and ordered feature list are loaded from `backend/ml/`. Keep these files together in the backend deployment. Notebook datasets, notebooks, and training-only dependencies are retained for reproducibility but are not needed to serve requests; a production image can omit `backend/ml/data/`, `backend/ml/notebooks/`, `backend/requirements-notebooks.txt`, and notebook-only packages after its build context is deliberately restricted. Do not omit the model ensemble or `pcos_model_features.json`.

## Configuration

- `NVIDIA_API_KEY`: required for natural-language feature extraction and question generation.
- `NVIDIA_MODEL`: NIM model name; defaults to `openai/gpt-oss-20b`.
- `NVIDIA_BASE_URL`: defaults to `https://integrate.api.nvidia.com/v1`.
- Intake follow-up questions are generated from local templates. Clear one-field yes/no and numeric responses are parsed locally; longer answers use the forced NVIDIA extraction tool.
- `CORS_ORIGINS`: comma-separated frontend origins; defaults to `http://localhost:5173`.
- `JWT_SECRET`: required random signing secret. Tokens expire after `JWT_EXPIRE_MINUTES` (default 60 minutes).
- `JWT_EXPIRE_MINUTES`: positive integer JWT lifetime; default `60`.
- `ADMIN_API_KEY`: static secret for account exports through the `X-Admin-Key` header. Keep it private and rotate it if exposed.
- `EMAIL_PROVIDER`: `smtp` (default) or `resend`. Use `resend` on Render Free, which blocks SMTP egress.
- `RESEND_API_KEY`, `EMAIL_FROM`: required with `EMAIL_PROVIDER=resend`. `EMAIL_FROM` must be a sender address on a verified Resend domain. Keep the key in the backend secret store only.
- `SMTP_USER`, `SMTP_APP_PASSWORD`: required with `EMAIL_PROVIDER=smtp`; the password must be a Gmail App Password. SMTP is suitable only for local development or a host that permits outbound SMTP.
- `ADMIN_NOTIFY_EMAIL`: required for the registration admin notice and website contact form; registration emails are sent to the new user and this admin address.
- If a provider is unavailable or a send fails, registration still succeeds and the failure is logged without credentials or submitted message contents. Contact form requests are accepted into FastAPI's background task queue; a `202` response means accepted for sending, not confirmed delivery.
- `TTS_PROVIDER`: `magpie` (default), `edge`, or `browser`. Magpie calls the NVIDIA hosted Magpie TTS Multilingual service from the authenticated backend; Edge TTS is also server-side; browser uses local Web Speech voices.
- `EDGE_TTS_VOICE`: Edge TTS voice name; defaults to `en-IN-NeerjaNeural`.
- `MAGPIE_TTS_URL`: NVIDIA hosted Magpie `/v1/audio/synthesize` endpoint; defaults to the public Magpie Multilingual invocation endpoint.
- `MAGPIE_TTS_VOICE`: NVIDIA voice name; defaults to `Magpie-Multilingual.EN-US.Aria`.
- `OSM_CONTACT_EMAIL`: optional real project contact included in the descriptive Nominatim User-Agent.
- `DATABASE_URL`: SQLite URL by default; the MVP creates `backend/pcos_agent.sqlite3` automatically.

Keep `.env` local and do not commit the key. Without an API key, the chat can produce deterministic question wording but cannot extract answers; configure NIM for a complete conversational intake.

## Data, notebooks, and training

Both notebooks read the workbook from `../data/PCOS_data_without_infertility.xlsx` and save model artifacts into their appropriate `ml/` locations. Execute them from `backend/ml/notebooks/` so those relative paths resolve. Install notebook dependencies with `pip install -r ../requirements-notebooks.txt` from that directory, then run:

```bash
jupyter nbconvert --to notebook --execute --output-dir /tmp/pcos-notebook-results pcos_random_forest_og_rf.ipynb
jupyter nbconvert --to notebook --execute --output-dir /tmp/pcos-notebook-results pcos_random_forest_smotenc.ipynb
```

The balanced notebook writes the canonical model and feature list. The SMOTENC notebook writes its model to `ml/experiments/` and writes the shared feature list. `backend/ml/train_pcos_model.py` remains the canonical baseline trainer and has not been changed. The runtime ensemble is built separately with `python ml/train_ensemble.py` from `backend/`; it writes five seeded models to `backend/ml/ensemble/` without replacing the baseline artifact or feature JSON. The new `11_Engineered_Features_Comparison.ipynb` reports why the existing baseline remains the recall-first recommendation; it excludes the reference notebook’s non-self-reportable fields.

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

Hip and waist measurements are asked during chat and can be skipped if unknown. The trained models require every input column, so an explicitly skipped/unavailable optional circumference and its ratio are represented as `0.0` at inference. This MVP assumption may affect the estimate. `/predict` expects `{ "features": { ...all 16 keys... } }` and returns the ensemble mean probability, `risk_label` (`elevated` at probability >= 0.5, otherwise `lower`), average feature importances, an agreement score, and individual probabilities for the five members. The probability is a model score, not a calibrated clinical risk unless separately validated.

When all intake fields are collected, `/chat` summarizes the answers and sets `awaiting_confirmation: true`. The backend does not derive BMI/waist-to-hip ratio or run inference at that point. It uses the extraction tool on the next user message so corrections can be merged; changed answers produce a fresh confirmation summary. An affirmative confirmation proceeds to Python derivation and the ensemble. Follow-up prompts list confirmed values and missing slots; invalid free-form questions fall back to a field-specific template. After 15 intake turns, the backend uses that direct single-field fallback.

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"age":25,"weight_kg":62,"height_cm":165,"cycle_length_days":35,"weight_gain":0,"hair_growth":0,"skin_darkening":0,"hair_loss":0,"acne":1,"fast_food":0,"regular_exercise":1,"hip_inch":38,"waist_inch":31,"cycle_irregular":1,"bmi":22.77,"waist_hip_ratio":0.816}}'
```

For `/chat`, send `{"message":"I am 25 years old"}` to `http://localhost:8000/chat`, then pass the returned `session_id` in each following request.

## Accounts, screening history, and research questions

Register with `POST /auth/register` using `{ "full_name": "...", "email": "...", "phone": "...", "password": "..." }`; login continues to use email and password. Email is normalized to lowercase, and phone spaces/dashes are removed while a leading `+` is retained. Passwords are stored as bcrypt hashes. The frontend keeps the JWT in localStorage, checks it with `GET /auth/me` on load, and sends `Authorization: Bearer <token>` on protected calls. Expired tokens return a distinct 401 response and the frontend signs the user out.

On startup, `init_db()` applies an additive SQLite migration for existing development databases: it adds `full_name` and `phone` columns with empty values for existing accounts and preserves their IDs, emails, password hashes, and screening rows. No database recreation is required. Existing accounts can continue signing in; they need a new registration/profile flow if they require profile values (no profile-edit endpoint is currently provided).

Registration queues an admin notification and a user welcome email using Gmail SMTP with STARTTLS. Missing settings or mail failures are logged without credentials and never roll back account creation. Emails contain profile details only; passwords and hashes are never included.

`GET /admin/users/export?format=csv` or `?format=xlsx` returns `full_name`, `email`, `phone`, and `created_at`, protected by `X-Admin-Key: $ADMIN_API_KEY`. Password hashes are excluded. The same data can be exported locally from `backend/` with `python scripts/export_users.py --format xlsx --output users.xlsx` (or `csv`).

The `/tts/config` endpoint tells the authenticated frontend which speech provider is configured. Magpie uses the existing `NVIDIA_API_KEY` to call NVIDIA's hosted multipart `/v1/audio/synthesize_online` API; the backend streams raw PCM chunks and the browser begins playback while later chunks are still arriving. The default English voice is `Magpie-Multilingual.EN-US.Aria`. Set `TTS_PROVIDER=edge` for Edge TTS or `browser` for local Web Speech synthesis (with an English voice picker saved in localStorage). Browser speech recognition remains the microphone input path. Hosted Magpie requires network access and NVIDIA API entitlement for that model.

Authenticated `/chat` conversations keep their live slot state in memory scoped by both user ID and session ID. When all model inputs are confirmed and the model ensemble completes, the result is persisted to the `screenings` table. `GET /screenings/me` returns the current user's completed results, newest first. Since live slot state is intentionally in memory for this MVP, an unfinished conversation does not survive a backend restart; completed screening history does.

After a screening, `POST /chat/research` accepts `{ "question": "..." }`. It retrieves relevant chunks from the curated markdown under `backend/app/agent/knowledge/`, then asks the configured NVIDIA model to answer using only those chunks. If retrieval finds no relevant material, the service declines rather than asking the model to answer from general knowledge. This provider is selected through a `ResearchProvider` interface; `KnowledgeBaseProvider` is the local implementation. It reads the latest saved screening only to personalize context and never calls, changes, or blends with the model prediction. Research replies include source references, screening-not-diagnosis language, and the persistent affiliation/general-information/licensed-care disclaimer. The documents cover overview, symptoms, possible contributors, lifestyle, when to seek care, and common misconceptions, with sources linked in each document.

`POST /doctors/nearby` remains available as a legacy backend endpoint for compatibility, but the current portal UI does not call it. The active doctor-finder UI opens Google Maps, Practo, and NMC externally instead. A location is sent only when the user opens a directory link. Any third-party directory data should be independently verified.

Example authenticated request:

```bash
curl -X POST http://localhost:8000/chat/research \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"What foods should I avoid?"}'
```

## Verification performed

- Audited source copies before changing anything. The original canonical pickle and feature JSON were byte-identical to the copies previously used by the backend; the SMOTENC artifact is a distinct pipeline.
- The two canonical training notebooks were previously executed from their relocated `ml/notebooks/` directory. The new engineered-feature comparison notebook was executed headlessly with `jupyter nbconvert`; it completed without cell errors and the output table is saved in the notebook.
- After restoring the original canonical pickle, loaded `backend/ml/pcos_rf_model.pkl` from its final location with the existing Python environment. Confirmed it is a `RandomForestClassifier` with `class_weight="balanced"`, 16 inputs, and that a complete sample row returns valid two-class `predict_proba` values.
- `python3 -m compileall` passed for the backend app and the new training script.
- The app imports and the modified modules pass `compileall`. The direct NVIDIA forced-tool request timed out at both 45 and 120 seconds without a response, so the live NIM extraction integration could not be confirmed in this environment.
- A prior mocked `/chat` smoke run covered sparse extraction and follow-ups before the confirmation gate was added. The current confirmation flow has a separate mocked authenticated integration run documented below; a live NIM integration still requires a reachable NVIDIA endpoint.
- `npm run build` passes after the auth, history, registration, and voice UI changes.
- Authenticated API integration was exercised with SQLite and the real canonical Random Forest. The registration/export/expiry checks, legacy SQLite migration, SMTP-mocked messages and failure handling, export CLI, protected routes, and previous full-chat/research tests passed.
- The sandbox did not provide browser speech voices, Gmail SMTP credentials/network, or the `edge-tts` package/service connection. Browser audio behavior, live mail delivery, and Edge TTS network playback remain to be checked locally.
- A pasted local Uvicorn trace showed successful auth/CORS followed by a 45-second NVIDIA read timeout in `next_question`, which escaped as an unhandled 500. Follow-up wording now has a configurable 15-second timeout and a safe local question fallback; a mocked `APITimeoutError` through `/chat` returned 200 and preserved the active session. The frontend now defaults to `127.0.0.1:8000` and reports actionable guidance when the API itself cannot be reached.


## Recent verification

- The recall comparison notebook was executed headlessly and its outputs are saved in the notebook. It compared the canonical baseline and two engineered candidates on a shared stratified split; baseline CV recall was 0.801 and all three tied on held-out recall (0.750), so the feature contract and baseline trainer remain unchanged.
- Trained and loaded five seeded models (`42`–`46`) from `backend/ml/ensemble/`; a real sample prediction returned individual probabilities, their mean, and agreement. A mocked authenticated three-turn `/chat` run verified no prediction before confirmation, an acne correction to false, derivation only after affirmative confirmation, and a single 16-feature model input.
- `python -m unittest discover -s tests -v` passes the false/zero slot regression, correction detection, affirmative classification, and missing-field follow-up checks. `npm run build` passed after the Narisaarthi brand and requested color tokens were applied.
- Phase 3 doctor finder and personalized research passed mocked service/route checks. Sandbox DNS could not resolve either OSM host (`curl` exit 6), so live Nominatim and Overpass requests still need verification on a network-enabled machine. The doctor-search UI and both visible disclaimers are present in the production build.

## Public website and contact

The public landing page is served at `/`; the existing login and screening portal is at `/app`. `/privacy` and `/legal` show the privacy policy and the dataset/licence notice. The public `POST /contact` endpoint validates the submitted name, email, subject, and message, then queues a notification to `ADMIN_NOTIFY_EMAIL` using the existing Gmail SMTP settings. It returns `503` with a setup message if SMTP is not configured. Do not submit clinical details through the general contact form.

The frontend includes a web app manifest and service worker and offers the browser install prompt when the browser reports that the site is installable. Installation requires HTTPS in deployment (localhost is allowed during development). On iOS, the page shows the “Share → Add to Home Screen” instructions.

The screening portal doctor finder now opens external directory searches directly: Google Maps first, followed by Practo and the official NMC doctor registry for credential checks. The UI no longer calls OpenStreetMap for doctor lookup. It can use browser geolocation or a manually entered city/pincode to compose searches; a location is sent to a directory only when the user opens its link. Practo's current API documentation describes an agreement-based partner API that issues credentials; no public free tier was verified, so the app does not assume API access or scrape Practo pages.

The legal notice links to the Kaggle PCOS dataset source and explicitly states that the original uploader's current licence and redistribution/commercial permissions have not been independently verified. Confirm those permissions before redistributing or commercially using the model/data.
