# Toolbox for Building Recommender Systems in Python

![CI/CD Workflow](https://github.com/tsmith5151/user-recommender/actions/workflows/ci.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/user-recommender/badge/?version=latest)](https://user-recommender.readthedocs.io/en/latest/?badge=latest)
___________

## Overview 

The objective of this project is to provide Data Scientist a framework which
can be easily be extended to rapidly develop recommendation systems. The idea
is to provide a series of core python classes using popular libraries like
[`Turi Create`](https://github.com/apple/turicreate) and
[`Tensorflow-Recommenders`](https://www.tensorflow.org/recommenders) that
provide a proxy for building recommenders using both explicit and implicit
data.

## Getting Started

For this project we will be leveraging [`poetry`](https://python-poetry.org/) to
handle installing  dependencies and versioning. This project requires `python > 3.7.1`.

To configure your environment, first initialize a virtual environment and then
install poetry: 

```
pip install poetry
```

Next, to install the required libraries you can simply run:

```
poetry install
```

Please refer to the series of helper notebooks in the *examples/* for examples
on incorporating the recommender classes into your own projects. For the demos,
feel free to use your own dataset or explore with the
(movielens)[https://grouplens.org/datasets/movielens/] dataset as well.
Currently, the examples are based off an anonymous online learning platform
dataset that contains thousands of user/item interactions. For more details,
please see the ".csv' files in *data/*. 

## Recommender Classes

- **CollaborativeFiltering** - A simple Collaborative Filtering class based on
 the following similarity distance metrics such as jaccard, cosine, and
 pearson. This class can be used to rank similar users or items based on the
 learned latent features. 

- **DeepMatrixFactorization** -  is an extension of Matrix Factorization for
 building recommendation engines using explicit data. The modification is to
 simply apply a non-linear kernel to model the latent feature interactions.

- **RankingFactorizationRecommender** - This class builds a Factorization
  Machine model to learn latent factors for each user and item interaction for
  implicit data (e.g. implicit Alternating Least Squares). As opposed to
  performing Matrix Factorization, we can now include slide features that may
  be numeric or categorical.

- **HybridRecommender** - one of the advantage of using neural networks for
 recommendation systems is the ability to create an architecture that utilizes
 both the collaborative and content based filtering approaches. This class
 exploits using explicit data to include side features for user/items.  Ideally
 this type of approach could help address the cold start problem or refined
 ranking given a subset of items filtered from a candidate generator model
 (e.g. see retrieval.py)

- **CandidateGeneration** - train a two-tower DNN model to produce the top 'k'
  recommendations by passing through the query tower and finding its
  representation from the learned embedding vector. An affinity score between
  the query and all the candidates is calculated and sorted with the k-nearest
  candidates to the query. The approach implemented is a brute-force method,
  but it's recommended to implement Approximate Nearest Neighbor (ANN), which
  is significantly faster.

## Model Deployment

If interested in deploying your model, a basic *Flask* example is provided in
*sever/* which can be used to make a RESTful api calls to a backend datastore.
This was just a quick prototype, but more to come on this section though and
example on serving a DNN model using [`Tensorflow-Serving`](https://www.tensorflow.org/tfx/guide/serving).  

## Contributing

If you are interested in contributing ideas on how to make this project better,
thoughts on other types of recommenders to include, or to simply report any
bugs, please feel free to open up an issue to discuss further. All
contributions are welcomed!  
