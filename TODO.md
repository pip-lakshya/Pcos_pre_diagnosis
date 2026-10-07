# Project status

## Completed

- [x] Add SQLite users/screenings tables, bcrypt password hashing, JWT register/login, and authenticated user resolution.
- [x] Protect chat, research, and screening history endpoints; scope active sessions by user ID.
- [x] Persist completed model results and expose the current user's screening history.
- [x] Add curated, sourced PCOS knowledge docs, local retrieval, research provider interface, grounded answer path, and safe out-of-scope fallback.
- [x] Keep research isolated from the Random Forest prediction and original risk explanation.
- [x] Add frontend login/register, token validation/storage, protected chat, follow-up research UI, and past screening list.
- [x] Verify register/login and invalid-password behavior, protected route 401s, full authenticated chat using the canonical model, database history, retrieval relevance, out-of-scope refusal, and frontend production build.
- [x] Add full name and normalized phone registration fields with safe additive migration for existing SQLite databases.
- [x] Add configurable JWT expiry and distinguish expired-token responses; expire frontend sessions on protected-route 401s.
- [x] Add background email delivery through Gmail SMTP or Resend HTTPS API, with safe logging and registration success preserved on mail failures.
- [x] Add admin-key protected CSV/XLSX exports and a matching local CLI export script without password hashes.
- [x] Add English browser voice selection, voice preference storage, sentence chunking/cancellation, and optional authenticated Edge TTS endpoint/provider selection.
- [x] Make NIM follow-up generation time bounded and recover to a local screening-safe question instead of an unhandled 500; align frontend's default backend URL with Uvicorn's IPv4 bind and show a useful connection message.
- [x] Verify valid/invalid registration, duplicate email, mocked email bodies and failure behavior, export formats/headers/auth, expired tokens, SQLite migration, export CLI, backend import, and frontend build.

## Still to do before production use

- [ ] Re-run direct NVIDIA extraction and generated research-answer checks when the NVIDIA NIM endpoint is reachable; direct requests timed out during this run.
- [ ] Configure a Resend API key and verified sender in Render, then verify real registration and contact email delivery. Mocked API tests do not confirm provider acceptance or delivery.
- [ ] Verify browser voice selection/STT/TTS on target browsers and Edge TTS playback with network access and `edge-tts` installed.
- [ ] Use HTTPS and a managed secret store in deployment; review localStorage token storage for the intended threat model.
- [ ] Decide whether unfinished screening sessions should persist across backend restarts.
- [ ] Have a qualified clinician review the user-facing health content and evaluate the screening model before any clinical use.
