PYTHON_VERSION=3.8.1
PROJECT = 'user-recommendation-system'

SHELL := /bin/bash

install:
	pip install poetry
	poetry install 

format:
	poetry run black .

lint:
	poetry run flake8

checks:
	make format
	make lint
