
## Data 
The dataset we will be working with can be accessed from the `data` directory
within the project root. Here's a breakdown of the available data:

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

