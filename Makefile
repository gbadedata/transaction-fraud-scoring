.PHONY: setup test lint format demo ieee-demo investigate figures ieee-figures dashboard dbt-build clean

setup:            ## install package + dev tools
	pip install -e ".[dev]"

test:             ## run the test suite
	pytest -q

lint:             ## static checks
	ruff check .

format:           ## auto-fix + format
	ruff check --fix .
	ruff format .

demo:             ## run the synthetic end-to-end pipeline demo
	python run_demo.py

ieee-demo:        ## run the IEEE-CIS pipeline (entity resolution + ring detection)
	python run_ieee_demo.py

investigate:      ## run the analyst investigation SQL over the CSVs (DuckDB)
	python scripts/run_investigation.py

figures:          ## regenerate the synthetic README figures
	python scripts/generate_figures.py

ieee-figures:     ## regenerate the IEEE-CIS figures
	python scripts/generate_ieee_figures.py

dashboard:        ## launch the Streamlit triage app (needs .[dashboard])
	streamlit run dashboards/streamlit_app.py

dbt-build:        ## build + test warehouse models (needs docker-compose up)
	cd dbt && dbt build

clean:
	rm -rf .pytest_cache **/__pycache__ dbt/target
