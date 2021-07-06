# User Remmendation System
___________

### Overview 
A sample of anonymized user data includes several variables representing users’
selected interests as well as how they interact with courses and assessments.
The objective is to leverage the dataset to calculate a score that represents
similarity between all active users. 

The result from the recommender application can be accessible via a RESTful
API. The endpoint will take a given user's unique ID and then return the top
'5' most similar users. Here we will be using Flask-RESTful, which is just an
extension for Flask that adds support for quickly building REST APIs in Python.
In addition, the back-end database SQLite.

The repository for this project can be cloned from on GitHub
[here](https://github.com/Tsmith5151/user-recommender).

### Data Overview: 
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

### Getting Started:

```
pip install -e requirements.txt
```

**Run User-Similarity Pipeline**

```
python -m pipeline --env dev --results_table user_ranking
```

*Set Environment Variables**

```
export DATABASE_ENV=dev
export RESULTS_TABLE=user_ranking
```

```
python -m runserver
```

### Summary

1. Tell us about your similarity calculation and why you chose it.

2. We have provided you with a relatively small sample of users. At true scale,
   the number of users, and their associated behavior, would be much larger.
   What considerations would you make to accommodate that?

3. Given the context for which you might assume an API like this would be used,
   is there anything else you would think about? (e.g. other data you would
   like to collect)
