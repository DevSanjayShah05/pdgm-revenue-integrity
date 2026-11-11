PY := .venv/bin/python
export PYTHONPATH := src:.

setup:       ## venv + pinned deps
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

reference:   ## re-download CMS files and rebuild data/reference (+ CMS test claims)
	./scripts/fetch_cms.sh && $(PY) scripts/build_reference.py

second-read: ## re-import OASIS error patterns from project 1 (path to its decisions.csv)
	$(PY) scripts/import_second_read_errors.py $(DECISIONS)

analyze:     ## simulate 20k periods, SQL summaries, isolated error impacts, QA model (~1 min)
	$(PY) -m period_audit.analyze

test:
	$(PY) -m pytest -q

api:
	.venv/bin/uvicorn api.main:app --reload --port 8001

dashboard:
	.venv/bin/streamlit run dashboard/app.py

.PHONY: setup reference second-read analyze test api dashboard
