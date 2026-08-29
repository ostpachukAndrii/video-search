.DEFAULT_GOAL := help
PY := .venv/bin/python

help:  ## показати доступні команди
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

venv:  ## створити оточення (легкі залежності, без ваг)
	python3.12 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -e ".[test]"

test:  ## прогін CI: тільки реалізоване, має бути зелено
	$(PY) -m pytest -q

wip:  ## backlog: показати, які сценарії ще не реалізовані
	$(PY) -m pytest -m wip -q --no-header -x --tb=line || true

offline:  ## довести п.9 — робота із заблокованою мережею
	$(PY) scripts/verify_offline.py

licenses:  ## довести п.6 — придатність ліцензій (ваги + залежності)
	$(PY) scripts/check_licenses.py
	$(PY) scripts/check_deps.py

doctor:  ## стан середовища: пристрій, ваги, набори
	PYTHONPATH=src $(PY) -m vsearch.cli doctor

golden:  ## перевірити цілісність золотого набору
	PYTHONPATH=src $(PY) -m vsearch.cli golden --show-missing

qdrant:  ## підняти Qdrant у Docker
	docker rm -f vsearch-qdrant 2>/dev/null || true
	docker run -d --name vsearch-qdrant -p 6333:6333 -p 6334:6334 \
		-v "$$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest
	@sleep 3 && curl -sf http://localhost:6333/healthz && echo " — Qdrant готовий"

smoke:  ## згенерувати синтетичний набір і проіндексувати його
	$(PY) scripts/make_smoke_set.py
	PYTHONPATH=src $(PY) -m vsearch.cli index tests/golden/smoke/media --recreate

compare:  ## A/B конфігурацій регіонів: make compare GOLDEN=real_photos
	$(PY) scripts/compare_configs.py --golden $(or $(GOLDEN),clutter)

ui:  ## підняти веб-інтерфейс для перевірки (http://127.0.0.1:7860)
	PYTHONPATH=src $(PY) -m vsearch.cli serve

demo:  ## наповнити індекс усіма синтетичними наборами і підняти інтерфейс
	$(PY) scripts/make_smoke_set.py
	PYTHONPATH=src $(PY) -m vsearch.cli index tests/golden/smoke/media --recreate
	PYTHONPATH=src $(PY) -m vsearch.cli index tests/golden/clutter/media
	PYTHONPATH=src $(PY) -m vsearch.cli index tests/golden/binding/media
	PYTHONPATH=src $(PY) -m vsearch.cli serve

models:  ## завантажити ваги (ПОТРІБНА МЕРЕЖА — лише етап збірки)
	$(PY) scripts/fetch_models.py

lock:  ## закріпити revision і sha256 у маніфесті (ПОТРІБНА МЕРЕЖА)
	$(PY) scripts/fetch_models.py --update-lock

image:  ## зібрати образ і довести офлайн-контур усередині нього
	docker build -t vsearch . && docker run --rm --network=none vsearch pytest -m offline -q

.PHONY: help venv test wip offline licenses doctor golden qdrant smoke compare ui demo models lock image
