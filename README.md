# Natural Language SQL Agent

Ask questions in plain English, get answers backed by a live MySQL query. This is a small Streamlit app that turns a natural-language question into SQL, runs it against a MySQL database, and returns the answer.

Built as a demo around a `t_shirts` inventory database (brand, color, size, price, stock, discounts), but the approach generalizes to any MySQL schema.

## How it works

1. **Few-shot retrieval** — a handful of hand-written `(question, SQL query, answer)` examples in [few_shots.py](few_shots.py) are embedded with a local sentence-transformers model and stored in a [Chroma](https://www.trychroma.com/) vector store. For each incoming question, the 2 most semantically similar examples are retrieved and used to steer SQL generation.
2. **SQL generation** — the question, retrieved examples, and database schema are assembled into a prompt (see [prompts.py](prompts.py)) and sent to Google's Gemini via `langchain-google-genai`.
3. **Execution** — the generated SQL runs against a local MySQL database.
4. **UI** — [main.py](main.py) is a one-page Streamlit app: type a question, see the answer.

All model/database wiring lives in [langchain_helper.py](langchain_helper.py).

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), a local MySQL server.

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Create the database:
   ```bash
   mysql -u root < database/db_creation_t_shirts.sql
   ```
   This creates the `tshirts` database with `t_shirts` and `discounts` tables and seeds them with randomized sample data. The app currently expects to connect as `root` with no password on `localhost` (see `get_db()` in [langchain_helper.py](langchain_helper.py)) — adjust there if your local MySQL setup differs.
3. Copy `.env.example` to `.env` and set your `GOOGLE_API_KEY` ([Google AI Studio](https://aistudio.google.com/)):
   ```bash
   cp .env.example .env
   ```
   `GOOGLE_MODEL` and `EMBEDDING_MODEL` have working defaults; override only if you want a different Gemini model or embedding model.
4. Run the app:
   ```bash
   uv run streamlit run main.py
   ```

## Notes

- The few-shot examples are embedded once and cached on disk under `chroma_store/` (git-ignored) — delete that folder if you edit `few_shots.py` and want the changes picked up.
- `.streamlit/config.toml` disables Streamlit's file watcher; edits to the app won't auto-reload — refresh the browser manually after changes.
