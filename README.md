# Toolbox for Recommender Systems  

![CI/CD Workflow](https://github.com/tsmith5151/user-recommender/actions/workflows/ci.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/user-recommender/badge/?version=latest)](https://user-recommender.readthedocs.io/en/latest/?badge=latest)
___________

## Overview 

The objective of this project is to build a toolbox than can be extended
further for rapidly developing recommendation systems. The two frameworks that
are currently utilized in this project for building the recommendation engines
are `Turi Create` and `Tensorflow`. 

## Examples

- Build a `Matrix Factorization` recommendation engine to return a collection of
  `k` most similar users whom's profile/interactions are similar to an input
  user `x`. 

- Train a [`two-tower DNN
  Retrieval Model`](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/6c8a86c981a62b0126a11896b7f6ae0dae4c3566.pdf)
  to recommend the top `k` items to an input user `x`.

## Dataset 
The dataset we'll build these examples from consists of anonymized user data captured
from an online platform that includes features such as user interests,
assessment scores, course viewing details, and tagged categories. Note the files
can be found in the `data` directory in the project root. 

Refer to the docs [here](https://user-recommender.readthedocs.io/en/latest/).
