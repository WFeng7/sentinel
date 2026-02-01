# Sentinel

Minimal setup to run the React + Tailwind frontend and FastAPI backend together.

## Prereqs

- Node.js 18+
- Python 3.10+

## Setup

1) Install root dev tooling (concurrently):

```bash
npm install
```

2) Install frontend deps:

```bash
npm --prefix frontend install
```

3) Create and activate a Python venv, then install backend deps:

```bash
python -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

## Run

From repo root:

```bash
npm run dev
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000/health
