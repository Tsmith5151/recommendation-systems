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
## Getting Started

Before getting started, the first step is to setup a virtual environment and
install the required Python dependencies. To install the required libraries,
run the following command:

```
pip install -r requirements.txt
```

Next, to run the `User-Similarity Pipeline`, the following command can be
executed from a command line:

```
python -m pipeline --env dev --results_table user_ranking
```

>**Note:** for a complete list of the input args and explanation, you can run
`python -m pipeline --help`. By default, the `env` argument is set to the
development environment. For production purposes, the `prod` flag can be
passed, however further configuration and database security features would need
to be incorporated. 

**RESTful API**

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
## Summary

1. Tell us about your similarity calculation and why you chose it.

2. We have provided you with a relatively small sample of users. At true scale,
   the number of users, and their associated behavior, would be much larger.
   What considerations would you make to accommodate that?

3. Given the context for which you might assume an API like this would be used,
   is there anything else you would think about? (e.g. other data you would
   like to collect)
