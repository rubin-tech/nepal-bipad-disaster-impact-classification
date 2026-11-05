# Project sources and run instructions

## Source materials provided for this project

1. `data/raw/Nepal_Disaster_Data_2020_2026.csv` — dataset included in the supplied project archive; use its data dictionary/column names for interpretation.
2. `STW7072CEM Machine Learning March Intake 2026 Coursework CW (Regular)-5f80e3c3-1cb9-45ca-9333-1b20458e0288 (1).pdf` — coursework brief supplied by the user (the brief requires a real-world classification/clustering application, application of three different techniques, Python/Jupyter or Colab code, and execution instructions).
3. `dashboard/app.py` — dashboard source from the supplied Nepal BIPAD project, patched so every Plotly chart has a unique Streamlit `key`.
4. `src/` — project data preparation and machine-learning implementation scripts included in the supplied archive.
5. `notebooks/` — Jupyter notebooks included in the supplied archive.

## Fix made
All `st.plotly_chart(...)` calls in `dashboard/app.py` now include distinct keys (`plotly_chart_1`, etc.) to prevent `StreamlitDuplicateElementId` when charts have identical parameters.

## Run
From the project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The assignment brief and original dataset are included in the project archive when available in the supplied materials. The coursework brief is a user-provided source, not a public external citation.
