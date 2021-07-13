
Similarity Metrics
******************

CF: Matrix Factorization
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: recommender.engine.similarity.RankingFactorizationRecommender
	:members:

.. autofunction:: recommender.engine.similarity.agg_similar_users_mf	


CF: Cosine Distance
~~~~~~~~~~~~~~~~~~~

.. autoclass:: recommender.engine.similarity.UserSimilarityMatrix
	:members:

.. autofunction:: recommender.engine.similarity.compute_weighted_matrix

.. autofunction:: recommender.engine.similarity.rank_similar_users