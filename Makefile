.PHONY: all data analysis sql app clean

all: data analysis sql

data:
	python scripts/generate_data.py

analysis:
	python scripts/run_analysis.py

sql:
	python scripts/run_sql_analysis.py

app:
	streamlit run app/streamlit_app.py

clean:
	rm -f data/raw/*.csv data/processed/*.csv data/processed/*.json analytics/*.duckdb models/*.joblib models/*.json

