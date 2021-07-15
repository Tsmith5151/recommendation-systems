## Background 

The objective of this project is to build an recommendation system which
returns a collection of users whom share similar profiles and interactions with
a given user of interest. The dataset consists of anonymized user data captured
from an online platform that includes features such as user interests,
assessment scores, course viewing details, and tagged categories. 

For this project, there were two methods explored for recommending users whom
share similar patterns. The first approach is a simplified version of
collaborative filtering where we generate an interaction matrix for all users
and the items they've interacted with (e.g. tags, view times, assessment
scores). We then build a user-item matrix and to compute a pairwise cosine
distance across all users and in return is utilized to rank the top users that
are most similar to a given user. The second approach consists of Factorization
Machines - which is basically a generalization of Matrix Factorization. This
approach learns a set of latent factors for each user and item and then uses
them to rank recommended users according to the likelihood of observing those
(user, item) pairs. This method produces the nearest users based on the cosine 
similarity between latent user factors.

The result from the recommender application can be accessible via a RESTful
API that is hosted locally (for now). The endpoint is based on the `Matrix
Factorization` method implemented and will take a given user's unique ID as the
input and return the top 'n' most similar users. For this project we
will be using Flask-RESTful, which is just an extension for Flask that adds
support for quickly building REST APIs in Python.

_______
## Data 
This document is accompanied by four csv files with the following tables:

**user_interests:**

| Keys                     | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| user_handle              | a unique identifier for each user                            | 
| interest_tag             | the he date the user viewed the course                       |
| date_followed            | the date user followed/expressed interest in the tag field   |


**user_assessment_scores:**


| Keys                     | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| user_handle              | a unique identifier for each user                                            | 
| user_assessment_date     | the date the user completed this assessment                                  |
| assessment_tag           | the assessment tag                                                           |
| user_assessment_score    | the user’s score                                                             |


**user_course_views:**

| Keys                     | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| user_handle              | a unique identifier for each user                                              | 
| view_date                | the he date the user viewed the course                                         |
| course_name              | name of course                                                                 |
| author_handle            | a unique identifier for each course’s author                                   |
| level                    | difficulty level assigned to this course                                       |
| course_view_time_seconds | number of seconds this user spent watching per day                             |


**course_tags:**

| Keys              | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| course_id         | course identifier for joining to user_course_views            | 
| course_tags       | author-applied tags to define what topics this course covers  |


______
## Requirements
- Python 3.6+

For this project, we will be leveraging [poetry](https://python-poetry.org/)
for our dependency manager. To get started, create a virtual environment and install poetry:

```
pip install poetry==1.1.7
```

Next, run the following command to install all dependencies from
`pyproject.toml`: 

```
poetry install
```

> **Note:** A quick step to configure a virtual environment and handling Python versioning
within this project is to simply run `make penv`. For more details on using
pyenv, check out the docs [here](https://github.com/pyenv/pyenv). Assuming
pyenv is now configured, you can then install the required dependencies into
the activated virtual environment by running:

```
make build-env
```

______
## Data Store

In terms of the backend database used for the application, `SQLite3` will be
the default choice. For further details on installing the SQLite3 database, 
please refer to the documentation [here](https://www.tutorialspoint.com/sqlite/sqlite_installation.htm).

______
## Run Pipeline

To run the `User-Similarity Pipeline`, the following command can be
executed from a command line within the project root dir:


**Method 1: Ranking Users based on Cosine Similarity**
```
python -m recommender.pipeline --env dev --method similarity --results_table user_ranking 
```

**Method 2: Ranking Users based on Matrix Factorization**

```
python -m recommender.pipeline --env dev --method factorization --results_table user_ranking 
```

The recommender pipeline can easily be scheduled as a job (e.g. airflow) in
order to frequently update the user similarity ranking table. Current benchmark
metrics for the pipeline to execute successfully with 10k users on a single
8-core 16GB CPU machine is roughly 10 minutes. 

>**Note:** for a complete list of the input args to execute the pipeline, you
can run `python -m recommender.pipeline --help`. By default, the `env` argument is set to
the development environment. For production purposes, the `prod` flag can be 
passed, however further configuration and database security features would need
to be incorporated. 

_______
## RESTful API

 For this application, we will leverage
 [Flask-RESTful](https://flask-restful.readthedocs.io/en/latest/), which is an
 extension for Flask that adds support for quickly building REST APIs in
 Python. Before running the Flask server, the following environment variables
 for configuring the database environment and results table will need to be set:
 
```
export DATABASE_ENV=dev
export RESULTS_TABLE=user_ranking
```

Once the above variables are set, we can then start Flask server by running the
following command: 

```
python -m server.run
```

>**Note:** the default settings is to host the Flask server on localhost and
listening on port 5000. To change this setting, you can modify `runserver.py`
by setting the host and port arguments in the `app.run` method, respectively. 

To make a API call to the hosted server and return a JSON object containing the
top `5` most similar users to the user of interest, the following CURL command
can be run:

```
curl -X GET -H "Content-type: application/json" \ 
-d "{\"user_handle\":\"110\"}" \
"http://0.0.0.0:5000/api/similarity/"
```

>**Note:** the user id for this example is `110`. Simply change the value of the 
user_handle key the above body field `-d` with a user id of interest. Keep in mind 
that the range of user ids is between 1 and 10000. 

Alternatively to using curl for making an API call, a Python utility function
can also be used to facilitate returning similar users to the provided input user 
id. The following command can be executed to return the same JSON object: 

```
python -m server.api --user_id 110
```

**Output JSON for Matrix Factorization Approach**

```json
[
   {
      "similar":"5537",
      "score":0.8957
   },
   {
      "similar":"7771",
      "score":0.8567
   },
   {
      "similar":"5843",
      "score":0.8519
   },
   {
      "similar":"239",
      "score":0.8512
   },
   {
      "similar":"9530",
      "score":0.8363
   },
   {
      "similar":"687",
      "score":0.7750
   },
   {
      "similar":"854",
      "score":0.7691
   },
   {
      "similar":"1199",
      "score":0.7563
   },
   {
      "similar":"9046",
      "score":0.7409
   },
   {
      "similar":"7341",
      "score":0.7401
   }
]
```

