PYTHON_VERSION=3.8.1
PROJECT = 'user-recommender'

SHELL := /bin/bash

pep8:
	poetry run black 

lint:
	poetry run flake8

precommits:
	make pep8
	make lint
