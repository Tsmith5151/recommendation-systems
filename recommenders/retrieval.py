# Standard two-tower Retrieval model using Tensorflow
# RetrievalModel can be extended for more complex
# architectures.
import os
import tempfile
import numpy as np
import tensorflow as tf
from typing import Dict, Text
import tensorflow_recommenders as tfrs
from tensorflow.keras.callbacks import EarlyStopping
from .utils.logger import get_logger

logger = get_logger(__name__)


class QueryModel(tf.keras.Model):
    def __init__(self, query: tf.Tensor, unique_user_id: int, embedding_dim=32):
        """
        Query Tower: build a set of layers that describe how raw user features
        to be transformed into numerical user representations. The steps
        below consist of converting the user ids into integer indices,
        then map those into learned embedding vectors. An additional
        preprocessing layer is applied to capture user viewing time. This
        framework can be expanded to include other similar preprocessed features.

        Parameters
        ----------
        query: tf.Tensor
            tensor of user representations.
        unique_user_id: tf.Tensor
            tensor containing user ids.
        embedding_dim: int
            dimensionality of the candidate representation; Higher values will
            correspond  to models that may be more accurate, but prone to
            overfitting.
        """

        super().__init__()

        # maps each raw values to unique integer
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=unique_user_id, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_user_id) + 1, embedding_dim),
            ]
        )

        self.user_viewing_normalization = (
            tf.keras.layers.experimental.preprocessing.Normalization()
        )

    def call(self, inputs):
        return tf.concat(
            [
                self.user_embedding(inputs["user_id"]),
                self.user_viewing_normalization(inputs["user_view_time"]),
            ],
            axis=1,
        )


class CandidateModel(tf.keras.Model):
    def __init__(
        self,
        candidate: tf.Tensor,
        unique_title_id: tf.Tensor,
        embedding_dim: int = 32,
        max_tokens: int = 1_000,
    ):
        """
        Candidate Tower: build a set of layers that describe how raw candidate
        features to be transformed into numerical userrepresentations.

        Preprocessing layer is applied to capture the fact
        that courses with very similar titles are likely to belong to the same
        series our courses; first step is to apply text tokenization (e.g.
        splitting words) and followed by vocabulary learning, then finally an
        embedding layer.

        Parameters
        ----------
        candidate: tf.Tensor
            tensor of candidate representations.
        unique_title_id: tf.Tensor
            tensor containing candidate ids.
        embedding_dim: int
            dimensionality of the candidate representation; Higher values will
            correspond  to models that may be more accurate, but prone to
            overfitting.
        max_tokens:int
            maximum tokens for text vectorization embedding layer
        """
        super().__init__()

        self.title_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=unique_title_id, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_title_id) + 1, embedding_dim),
            ]
        )
        self.title_vectorizer = (
            tf.keras.layers.experimental.preprocessing.TextVectorization(
                max_tokens=max_tokens
            )
        )

        self.title_text_embedding = tf.keras.Sequential(
            [
                self.title_vectorizer,
                tf.keras.layers.Embedding(max_tokens, embedding_dim, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.title_vectorizer.adapt(candidate)

    def call(self, titles):
        return tf.concat(
            [
                self.title_embedding(titles),
                self.title_text_embedding(titles),
            ],
            axis=1,
        )


class CandidateGeneration(tfrs.models.Model):
    def __init__(self, query, candidate, unique_user_ids, unique_title_ids):

        """
        Candidate generator class to build a two-tower retrieval model

        Loss Function: maximizes the predicted user-courses affinity for
        watches observed, and minimizes it for watches that did not happen.
        The model output is the dot product between the user_id embedding
        and the item_id embedding.

        Parameters
        ----------
        query: tf.Tensor
            tensor of query representations.
        candidate: tf.Tensor
            tensor of candidate representations.
        unique_user_id: tf.Tensor
            tensor containing user ids.
        unique_title_id: tf.Tensor
            tensor containing candidate ids.
        """
        super().__init__()

        self.query_model = tf.keras.Sequential(
            [QueryModel(query, unique_user_ids), tf.keras.layers.Dense(32)]
        )
        self.candidate_model = tf.keras.Sequential(
            [CandidateModel(candidate, unique_title_ids), tf.keras.layers.Dense(32)]
        )

        # factorized top k metrics
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=candidate.batch(128).map(self.candidate_model)
        )

        self.task = tfrs.tasks.Retrieval(metrics=metrics)

    def compute_loss(self, inputs: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        query_embeddings = self.query_model(
            {
                "user_id": inputs["user_id"],
                "user_view_time": inputs["user_view_time"],
            }
        )
        positive_candidate_embeddings = self.candidate_model(inputs["title"])

        return self.task(query_embeddings, positive_candidate_embeddings)


def retrieve_topk_candidates(
    candidate: tf.Tensor,
    model: tf.keras.Sequential,
    batch_size: int,
    k: int,
) -> None:
    """
    Generate top 'k' recommendations by passing through the query tower and find
    its representation from the learned embedding vector. An affinity score
    between the query and all the candidates is calculated and sorted with
    the k-nearest candidates to the query. The approach implemented is a
    brute-force method, but it's recommended to implement Approximate
    Nearest Neighbor (ANN), which is significantly faster.

    Parameters
    ----------
    candidate: tf.Tensor
        input candidate dataset
    model: tf.keras.Sequential
        trained retrieval model
    batch_size:int
        number of samples per batch of computation.
    k: int
        number of items to retrieve for recommending to users
    """
    brute_force = tfrs.layers.factorized_top_k.BruteForce(
        model.query_model.layers[0].layers[0], k=k
    )
    brute_force.index(
        candidate.batch(batch_size).map(model.candidate_model.layers[0].layers[0]),
        candidate,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        retrieval_model_path = os.path.join(tmp_dir, "retrieval_model")

    brute_force(tf.constant([""]))  # build layer
    brute_force.save(
        retrieval_model_path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["BruteForce"]),
    )


def train_test_split(data: tf.Tensor, train_ratio: int, batch_size: int) -> tf.Tensor:
    """
    Function to split data into train/test. Note: in an practice we would
    most likely be done by time: the data up to time  would be used to
    predict interactions.

    Parameters
    ----------
    data: tf.Tensor
        input query dataset
    train_ratio: int
        percentage to split train/test
    """
    train_split_size = train_ratio * data.__len__().numpy()

    # shuffle the elements of the dataset randomly.
    shuffled = data.shuffle(buffer_size=10_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(train_split_size)
    test = shuffled.skip(train_split_size)
    logger.info(f"Train Size: {train.__len__()} -- Test Size: {test.__len__()}")

    cached_train = train.shuffle(1_000).batch(batch_size)
    cached_test = test.batch(batch_size).cache()
    return cached_train, cached_test


def retrieval_main(
    query: tf.Tensor,
    candidate: tf.Tensor,
    epochs: int = None,
    learning_rate: int = 0.10,
    batch_size: int = 5_000,
    split_ratio: float = 0.80,
    top_k: int = 10,
):
    """
    Retrieval Model: Helper function to generate user/content vocabulary;
    split train/test dataset, train/evaluate two-tower retrieval model,
    and finally generate topk recommended items for an input query.

    Parameters
    ----------
    query: tf.Tensor
        input query dataset
    candidate: tf.Tensor
        input candidate dataset
    epochs: int
        number of epochs to train the model. An epoch is an iteration over the
        entire x and y data provided.
    learning_rate:int
        amount of change to the model during each step of this search process
    batch_size:int
        number of samples per batch of computation.
    train_ratio: int
        percentage to split train/test
    top_k: int
        number of items to retrieve for recommending to users
    """

    # maps each raw values to unique integer
    unique_user_ids = np.unique(
        np.concatenate(list(query.batch(batch_size).map(lambda x: x["user_id"])))
    )
    unique_title_ids = np.unique(np.concatenate(list(candidate.batch(batch_size))))
    logger.info(
        f"Unique Users: {len(unique_user_ids)} -- Unique Titles: {len(unique_title_ids)}"
    )

    # split train/test and cache
    cached_train, cached_test = train_test_split(
        query, train_ratio=split_ratio, batch_size=batch_size
    )

    # Build Retrieval Model
    model = CandidateGeneration(query, candidate, unique_user_ids, unique_title_ids)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

    # fit model
    logger.info("Train Retrieval Model...")
    early_stop = EarlyStopping(
        monitor="val_factorized_top_k/top_100_categorical_accuracy",
        mode="auto",
        verbose=0,
        patience=3,
    )
    history = model.fit(
        cached_train,
        validation_data=cached_test,
        validation_freq=1,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1,
    ).history

    logger.info("Evaluating Retrieval Model...")
    model.evaluate(cached_test, return_dict=True)

    logger.info("Generate Top-K Recommendations for Users...")
    retrieve_topk_candidates(
        candidate,
        model,
        batch_size=batch_size,
        k=top_k,
    )
    return model, history
