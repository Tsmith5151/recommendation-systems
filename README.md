# User Remmendation System
___________

## Overview 
A sample of anonymized user data includes several variables representing users’
selected interests as well as how they interact with courses and assessments.
The objective is to leverage the dataset to calculate a score that represents
similarity between all active users. 

The result from the recommender application can be accessible via a RESTful
API. The endpoint will take a given user's unique ID and then return the top
'5' most similar users. Here we will be using Flask-RESTful, which is just an
extension for Flask that adds support for quickly building REST APIs in Python.
In addition, the back-end database SQLite.
_______
## Data Overview
This document is accompanied by four csv files with the following tables:

**user_assessment_scores**

- *user_handle*: a unique identifier for each user, consistent across tables 
- *user_assessment_date*: the date the user completed this assessment
- *assessment_tag*: the assessment tag (may or may not exactly match the
user_interestor course_view tags)
- *user_assessment_score*: the user’s score, from a distribution that is
roughly normal with mean of ~140 and a standard deviation of ~60

**user_course_views**

- *user_handle*: a unique identifier for each user 
- *view_date*: the date the user viewed the course
- *course_name*: self-explanatory
- *author_handle*: a unique identifier for each course’s author
- *level*: the difficulty level assigned to this course
- *course_view_time_seconds*: the number of seconds this user spent watching
 this course on this day

**course_tags**

- *course_id*: course identifier for joining to user_course_views
- *course_tags*: author-applied tags to define what topics this course covers
- *user_handle*: a unique identifier for each user
- *interest_tag*: tags this user has indicated they are interested in 
- *date_followed*: the date this user followed/expressed interest in the tag
mentioned above.

______
## Configure Environment 

Before getting started, the first step is to setup a virtual environment and
install the required Python dependencies. To install the required libraries,
run the following command:

```
pip install -r requirements.txt
```

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
  
2.) Apply Data Preprocessing
- Remove missing values
- Standardize column renaming
- Create and encode categorical features:
   - Bin assessment scores into quantiles: high, medium, low
   - Bin user content viewing time into quantiles: high, medium, low, very low
   - Convert interest, course, and assessment raw tags to encodings 
 
3.) Prepare a user-content matrix (e.g. users x feature) for each table. 

4.) Apply TruncatedSVD to reduce the high dimensionality feature space to a set
of lower dimensions that explains 90% of the total variance of the dataset. 

5.) Compute pairwise cosine similarity for each table

6.) Ensemble user, assessment, and course_tags tables  into one matrix
(n_users X n_users). 
- Each table is assigned a weight in order to control which table(s) are more
  influential when aggregating the three tables. 
    
7.) Rank top 5 most similar users per each unique user id.

8.) Write dataframe to a user ranking table in the SQLite3 database

The recommender pipeline can easily be scheduled as a job (e.g. airflow) in
order to frequently update the user similarity ranking table. Current benchmark
metrics for the pipeline to execute successfully with 10k users on a single
8-core 16GB CPU machine is roughly 10 minutes. 

>**Note:** for a complete list of the input args to execute the pipeline, you can run
`python -m pipeline --help`. By default, the `env` argument is set to the
development environment. For production purposes, the `prod` flag can be
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

1. Tell us about your similarity calculation and why you chose it.

2. We have provided you with a relatively small sample of users. At true scale,
   the number of users, and their associated behavior, would be much larger.
   What considerations would you make to accommodate that?

3.) Improvement to the API:
- The current API returns a JSON payload containing only the user ID and
score, which by default is the cosine similarity measurement. Future work
for the application could consist of including additional meta data which
would provide further context into explaining why the following users were
recommended. Meta information could include tags, course IDs, assessment
scores, and content viewing time. In the current scenario, the user ranking
table is being overwritten with the results from the most recent run.
Including a field such as a timestamp would also allow us to achieve 
previous runs in the event we would like to examine how the recommender
evolves over time for a given user.
