PROJECT = 'user-recommendation-system'
SHELL := /bin/bash
EXEC := poetry run
SPHINX_AUTO_EXTRA:=

install:
	pip install poetry
	poetry install 

format:
	@${EXEC} black .

format-check:
	@${EXEC} black . --check

lint:
	@${EXEC} flake8 .

doc:
	@${EXEC} sphinx-autobuild -b html docs docs/build/htm ${SPHINX_AUTO_EXTRA}

checks:
	make format-check
	make lint
