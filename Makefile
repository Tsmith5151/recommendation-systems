PYTHON_VERSION=3.8.1
PROJECT = 'user-recommender'

SHELL := /bin/bash

config-venv:
	git clone https://github.com/pyenv/pyenv.git ~/.pyenv
	@echo export PYENV_ROOT=$HOME/.pyenv >> ~/.bashrc
	@echo export PATH=$PYENV_ROOT/bin:$PATH >> ~/.bashrc
	git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
	@echo eval $(pyenv init --path) >> ~/.bashrc
	@echo eval $(pyenv virtualenv-init -) >> ~/.bashrc
	source ~/.bashrc

build-env:
	pyenv install $(PYTHON_VERSION) --force 
	pyenv virtualenv $(PYTHON_VERSION) $(PROJECT)
	pyenv local $(PROJECT)
	pip install -r requirements.txt

pep8:
	python -m black recommender/ 
	python -m black server/ 
	python -m black pipeline.py 
	python -m black parse.py 

lint:
	flake8 server --ignore=E501
	flake8 recommender/ --ignore=E501
	flake8 pipeline.py --ignore=E501
	flake8 parse.py --ignore=E501

precommits:
	make pep8
	make lint
