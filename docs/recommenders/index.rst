Recommender Algorithms
**********************

Simple Recommender (e.g. Cosine Similarity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: recommenders.simple.SimpleRecommender
	:members:
    
    
Matrix Factorization
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: recommenders.mf.MatrixFactorization
	:members:


Factorization Machines
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: recommenders.fm.RankingFactorizationRecommender
	:members:

Deep Neural Networks
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: recommenders.dnn.QueryModel
	:members:
    
.. autoclass:: recommenders.dnn.CandidateModel
	:members:
    
.. autoclass:: recommenders.dnn.RetrievalModel
	:members:
    
.. autofunction:: recommenders.dnn.train_test_split

.. autofunction:: recommenders.dnn.retrieve_topk_candidates

.. autofunction:: recommenders.dnn.retrieval_main

