# AI Data Analyst

Upload a CSV or Excel file: the app cleans it, analyses it, draws charts,
and answers questions about it with a local AI model (Llama in Ollama).

## Run

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
streamlit run app.py
```

## Files

| File | What it does |
|------|--------------|
| `app.py` | The page (Streamlit). Reads top to bottom: upload → overview → cleaning → analysis → AI. |
| `src/data_loader.py` | Reads the file and cleans it (text → numbers / dates, duplicates, empty columns). |
| `src/analysis.py` | Finds column types and builds the tables (KPIs, monthly trend, categories) and the summary sent to the AI. |
| `src/questions.py` | Answers simple questions with pandas, no AI ("total Sales", "Profit by Region", "top 5 ..."). |
| `src/ai.py` | Talks to Ollama: insights report, AI-chosen calculations, free answers. |
| `src/document_engine.py`, `src/knowledge_base.py` | PDF reading for a future feature (not used yet). |

## How a question is answered

1. `questions.py` tries to answer it directly with pandas.
2. If it can't, the AI **chooses** a calculation and pandas runs it (exact numbers).
3. If it's an open question, the AI answers from the dataset summary.
