.PHONY: setup test lint format demo dashboard dbt-build clean

setup:            ## install package + dev tools
	pip install -e ".[dev]"

test:             ## run the test suite
	pytest -q

lint:             ## static checks
	ruff check .

format:           ## auto-fix + format
	ruff check --fix .
	ruff format .

demo:             ## run the end-to-end pipeline demo
	python run_demo.py

figures:          ## regenerate the README figures
	python scripts/generate_figures.py

dashboard:        ## launch the Streamlit triage app (needs .[dashboard])
	streamlit run dashboards/streamlit_app.py

dbt-build:        ## build + test warehouse models (needs docker-compose up)
	cd dbt && dbt build

clean:
	rm -rf .pytest_cache **/__pycache__ dbt/target
