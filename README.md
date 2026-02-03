# Sentinel

Sentinel is an AI system that turns live traffic camera footage into real-time incident alerts and structured reports. Instead of waiting for someone to call 911 or manually monitor dozens of screens, Sentinel continuously watches roadway video, detects abnormal patterns like sudden stops or collisions, and determines whether an accident has likely occurred. When it detects a high-risk event, it automatically analyzes the scene, generates a structured incident summary, and sends it to operators for immediate dispatch, saving critical minutes when every second can mean the difference between life and death.

<img width="2052" height="920" alt="image" src="https://github.com/user-attachments/assets/70b88342-7bfb-431e-95e6-cbd540cb0567" />

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
