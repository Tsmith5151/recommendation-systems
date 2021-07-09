PYTHON_VERSION=3.8.1
PROJECT = 'user-recommender'

SHELL := /bin/bash

format:
	poetry run black .

lint:
	poetry run flake8

precommits:
	make format
	make lint
