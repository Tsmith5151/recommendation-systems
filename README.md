# User Remmendation System
___________

## Overview 

The objective of this project is to build an recommendation system which
returns a collection of users whom share similar profiles and interactions with
a given user of interest. The dataset consists of anonymized user data captured
from an online platform that includes features such as user interests,
assessment scores, course viewing details, and tagged categories. 

For this project, we will build a recommender system to first generate a
user-item matrix to compute a pairwise similarity metric across all users and
in return is utilized to rank the top users that are most similar to a given
user. The result from the recommender application can be accessible via a RESTful
API that is hosted locally (for now). The endpoint will take a given user's
unique ID and then return the top '5' most similar users. For this project we
will be using Flask-RESTful, which is just an extension for Flask that adds
support for quickly building REST APIs in Python.

_______
## Data Overview
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

The required dependencies for the project can be installed by running: 

```
pip install -r requirements.txt
```

Alternatively, [penv](https://github.com/pyenv/pyenv) can be installed and
configured for managing Python versioning and virtual environments by running:

```
make penv
```

To install the required dependencies into the activated virtual environment you
can run:

```
make build-env
```

In terms of the backend database used for the application, `SQLite3` will be
the default choice. For further details on installing the SQLite3 database, 
please refer to the documentation [here](https://www.tutorialspoint.com/sqlite/sqlite_installation.htm).

______
## Run Recommendation Pipeline

To run the `User-Similarity Pipeline`, the following command can be
executed from a command line:

```
python -m pipeline --env dev --results_table user_ranking
```

The user recommender pipeline executes the following steps:

1.) Load user interest, user assessment, user views, and course tags data from
  the respected tables in a SQLite3 backend database. 
- Tables:
  - user_interest
  - user_assessment_scores
  - user_courses
  - course_tags
  
2.) Apply data preprocessing.
- Remove missing values
- Standardize column renaming
- Create and encode categorical features:
   - Bin assessment scores into quantiles: high, medium, low
   - Bin user content viewing time into quantiles: high, medium, low, very low
   - Convert interest, course, and assessment raw tags to encodings 
 
3.) Prepare a user-content matrix (e.g. users x feature) for each user/course table. 

4.) Apply TruncatedSVD to reduce the high dimensionality feature space to a set
latent features of lower dimensions that explains 90% of the total variance of
the dataset. 
- The explained variance threshold is tunable and can be increased to
  include additional latent features. To reduced extra compute time for now,
  the variance threshold is set at 90%. 

5.) Compute pairwise cosine similarity for each table.

6.) Ensemble user, assessment, and course_tags tables  into one matrix
(n_users X n_users). 
- Each table is assigned a weight in order to control which table(s) are more
  influential when aggregating the three tables. 
- If a user ID is not in the assessment or courses table, the corresponding
  features will be zero for the user row. Therefore, the assessment or course
  table will not have an influence in the user-similarity and ranking matrix. 
    
7.) Rank top 5 most similar users per each unique user id.

8.) Write dataframe to a user ranking table in the SQLite3 database.

The recommender pipeline can easily be scheduled as a job (e.g. airflow) in
order to frequently update the user similarity ranking table. Current benchmark
metrics for the pipeline to execute successfully with 10k users on a single
8-core 16GB CPU machine is roughly 10 minutes. 

>**Note:** for a complete list of the input args to execute the pipeline, you
can run `python -m pipeline --help`. By default, the `env` argument is set to
the development environment. For production purposes, the `prod` flag can be 
passed, however further configuration and database security features would need
to be incorporated. 

_______
## RESTful API Service

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
python -m runserver
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
python -m server.similarity --user_id 110
```

**Output JSON**

```json
{
  "110": [
    "{'user': 5182, 'score': 0.4331}", 
    "{'user': 5029, 'score': 0.4159}", 
    "{'user': 1133, 'score': 0.4083}", 
    "{'user': 7477, 'score': 0.4043}", 
    "{'user': 2867, 'score': 0.3992}"
  ]
}
```
______
## Analysis and Future Work

**1.) Similarity Calculation**

The similarity metric utilized in the user-recommender application for pairwise
comparison is `cosine`. This method computes the cosine of the angle between
two vectors that are projected in a multidimensional space. In our case,
dependant on the table, features are derived from the tags, assessment scores,
and experience level fields. These features are all categorical and are encoded
into unique indexes. Cosine Similarity is a value that is bound by a
constrained range of 0 and 1. For example, if two vectors are exactly the same,
the similarity between the two is 1. Therefore, the higher the measurement
between the two vectors, the more similar they are. 

The design of our user-recommender is a form of collaborative filtering. In
particular, the design is a user-item filtering approach where we attempt to
find users that are similar to given user based on having similar interactions.
These interactions involve user interests, course viewing details, and user
assessment information. One drawback from using this approach is that it will
not scale to larger datasets. In order to compute similarities, the entire
dataset needs to be loaded into memory. One solution would be to convert from a
dense to a sparse matrix representation given that many of the rows have zero
entries. Using implementation `sparse.csr_matrix` can be used as it's very
memory efficient for storing sparse datasets. There would be minimal changes to
the code as `sklearn.metrics.pairwise.cosine_similarity` supports a using
sparse matrix directly. Secondly, collaborative filtering methods also do not
work well when a new user is added and has minimal information (e.g. cold-start
problem).

One approach to help with scaling is applying matrix factorization (MF). This
method helps decompose a high dimensional feature space into a set of latent
features. One common MF method is Singular value decomposition (SVD). In our
case, truncatedSVD was applied and a minimum variance explained threshold
parameter was set to return a series of latent features with lower dimensions.
One observation to note at this stage is SVD can be very slow and expensive to
compute. It would be recommended to explore other options such applying
Alternating Least Square (ALS), in which the matrix factorization can computed
in parallel. 

**2.) Scalability**

One thing to keep in mind when developing recommendation systems is
scalability. In this application, the dataset is relatively small sample with
only 10k users. In practice, we could expect millions of users and the current
design would need to be optimized. First, we could design a segmentation model
to break-up computing a single monolithic pairwise similarity matrix. In other
words, the first task could be to use the user/course attributes to create a
set of features vectors, which are inputs into a clustering algorithm (e.g.
K-Means) in order to segment similar users into 'k' clusters. Then we could
apply a pairwise similarity metric for all customers assigned to a given
cluster across multiple nodes. This approach allows the user similarity matrix
to be constructed and results written to the data store in parallel. Moreover,
we could also convert the data preprocessing steps in `recommender/utils` to
PySpark for additional parallelization. 


**3.) Improvement to the API**

The current API returns a JSON payload containing only the user ID and score,
which by default is the cosine similarity measurement. Future work for the
application could consist of including additional meta data which would provide
further context into explaining why the following users were recommended. Meta
information could include tags, course IDs, assessment scores, and content
viewing time. In the current scenario, the user ranking table is being
overwritten with the results from the most recent run.  Including a field such
as a timestamp would also allow us to achieve previous runs in the event we
would like to examine how the recommender evolves over time for a given user.
Finally, if additional meta information were to be included, a document based
data store would be recommended for retrieving semi-structured data.
