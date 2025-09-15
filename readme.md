# InsightAI

**Intelligent operator comment analysis system with production event context**

---

## Description

InsightAI is a system that analyzes operator text comments, links them to production events, and provides valuable insights using LLM and RAG pipeline. The system allows for quick extraction of actionable information and integration into applications via API.

---

## Key features

- Operator comment analysis
- Contextual response generation via Google Gemini API
- Comment storage in PostgreSQL
- Vector data representation in Chroma DB for fast search
- API for integration with external applications
- Batch processing of new comments and checking data relevance
- Function testing using pytest

---

## Installation and run (local)

1. Clone the repository:
```bash
git clone <repo_URL>
cd InsightAI
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate # Windows
pip install -r requirements.txt
```

3. Run PostgreSQL locally (or via Docker):

```bash
docker run -d -p 5432:5432 --name mypostgres -e POSTGRES_PASSWORD=yourpassword postgres:16
```

4. Set up the database connection in `utils/db.py`.

5. Start FastAPI server:

```bash
uvicorn main:app --reload
```

6. Open API documentation in a browser:

```
http://127.0.0.1:8000/docs
```

---

## Project structure

```
InsightAI/
│
├─ main.py # Entry point
├─ requirements.txt
├─ models/ # ORM models for PostgreSQL
├─ routes/ # FastAPI routes
├─ pipeline/ # Data processing and RAG
├─ utils/ # Helper functions (connection to DB)
└─ .venv/ # Virtual environment
```

---

## Testing

* Tests are in the `tests/` folder.
* Run tests:

```bash
pytest
```

---

## Prospects

* Deploy to the cloud (Cloud Run / Azure)
* Connect additional data sources
* Expanding analytics functionality

