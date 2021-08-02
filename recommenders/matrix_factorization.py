import numpy as np
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    Dot,
    Reshape,
    Flatten,
    Input,
    Embedding,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from typing import List
from .utils.logger import get_logger

logger = get_logger(__name__)


class MatrixFactorization:
    def __init__(
        self,
        unique_users: np.ndarray,
        unique_items: np.ndarray,
        embedding_dim: int = None,
        epochs: int = None,
        batch_size: int = None,
    ):
        """
        Class supporting Generalized Matrix Factorization. GMF exploits
        explicit feedback in by applying an inner product on the latent
        features of users and item.

        Parameters
        ----------
        n_users: np.ndarray
            input query dataset
        n_items: np.ndarray
            input candidate dataset
        epochs: int
            number of epochs to train the model. An epoch is an iteration over the
            entire x and y data provided.
        batch_size:int
            number of samples per batch of computation.
        embedding_dim: int
            number of dimensions for embedding layer
        """

        self.unique_users = unique_users
        self.unique_items = unique_items
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.loss = "mse"

    def __repr__(self):
        return """ Matrix Factorization Recommendation Engine """

    def build_model(self, x, y):
        """Build Deep Matrix Factorization Model

        Parameters
        ----------
        x: np.ndarray
           input training data; example input: [use_id,item_id,features]
        y: np.ndarray
           input target; example movie ratings
        """

        # User Embeddings
        self.user_id_input = Input(shape=[1], name="user")
        self.user_embedding = Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.unique_users.shape[0],
            input_length=1,
            embeddings_regularizer=l2(1e-6),
            name="user_embedding",
        )

        # Item Embeddings
        self.item_id_input = Input(shape=[1], name="item")
        self.item_embedding = Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.unique_items.shape[0],
            input_length=1,
            embeddings_regularizer=l2(1e-6),
            name="item_embedding",
        )

        # Flatten the embedding vector as latent features in GMF
        self.user_latent = Flatten()(self.user_embedding(self.user_id_input))
        self.item_latent = Flatten()(self.item_embedding(self.item_id_input))
        self.output = Dot(1, normalize=False)([self.user_latent, self.item_latent])

        self.model = Model(
            inputs=[self.user_id_input, self.item_id_input], outputs=self.output
        )
        self.model.compile(loss=self.loss, optimizer="adam")

    def _evaluate(self, X, Y, name):
        """Evaluate Model"""
        mse = self.model.evaluate(X, Y)
        logger.info(f"{name} mean squared error: {mse:.4f}")

    def train(self, X_train: np.ndarray, Y_train: np.ndarray):
        """Helper function to train model

        Parameters
        ----------
        x: np.ndarray
           input training data; example input: [use_id,item_id,features]
        y: np.ndarray
           input target; example movie ratings
        """

        self.build_model(X_train, Y_train)
        early_stop = EarlyStopping(
            monitor="val_loss", mode="auto", verbose=0, patience=5
        )
        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            shuffle=True,
            verbose=1,
            callbacks=[early_stop],
        ).history

        self._evaluate(X_train, Y_train, "train")
        return history


class DeepMatrixFactorization:
    def __init__(
        self,
        unique_users: np.ndarray,
        unique_items: np.ndarray,
        embedding_dim: int = None,
        epochs: int = None,
        batch_size: int = None,
        layers: List[int] = None,
        dense_units: int = None,
        dropout: float = None,
    ):
        """
        This class is an extension of Matrix Factorization for building
        recommendation engines using explicit data. The modification is to
        simply apply a non-linear kernel to model the latent feature
        interactions.

        Parameters
        ----------
        n_users: np.ndarray
            input query dataset
        n_items: np.ndarray
            input candidate dataset
        epochs: int
            number of epochs to train the model. An epoch is an iteration over the
            entire x and y data provided.
        layers: List[int]
            input list of layers, where the index is the size of each layer
        dropout:int
            randomly sets input units to 0 with a frequency of rate at each
            step during training time to help prevent overfitting
        batch_size:int
            number of samples per batch of computation.
        embedding_dim: int
            number of dimensions for embedding layer
        """

        self.unique_users = unique_users
        self.unique_items = unique_items
        self.layers = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.loss = "mse"

    def __repr__(self):
        return """ Deep Matrix Factorization Recommendation Engine """

    def build_model(self, x, y):
        """Build Deep Matrix Factorization Model

        Parameters
        ----------
        x: np.ndarray
           input training data; example input: [use_id,item_id,features]
        y: np.ndarray
           input target; example movie ratings
        """

        # User Embeddings
        self.user_id_input = Input(shape=[1], name="user")
        self.user_embedding = Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.unique_users.shape[0],
            input_length=1,
            embeddings_regularizer=l2(1e-6),
            name="user_embedding",
        )(self.user_id_input)
        self.user_vector = Reshape([self.embedding_dim])(self.user_embedding)

        # Item Embeddings
        self.item_id_input = Input(shape=[1], name="item")
        self.item_embedding = Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.unique_items.shape[0],
            input_length=1,
            embeddings_regularizer=l2(1e-6),
            name="item_embedding",
        )(self.item_id_input)
        self.item_vector = Reshape([self.embedding_dim])(self.item_embedding)

        # Concat user/item vectors
        self.vector = Concatenate()([self.user_vector, self.item_vector])

        # Add dense layers
        for idx in range(0, len(self.layers)):
            layer = Dense(
                self.layers[idx],
                kernel_regularizer="l2",
                activation="relu",
                name=f"layer{idx+1}",
            )(self.vector)
            self.vector = Dropout(self.dropout)(layer)

        # Output layer
        self.output = Dense(1, name="prediction")(self.vector)

        # Defined model
        self.model = Model(
            inputs=[self.user_id_input, self.item_id_input], outputs=self.output
        )
        self.model.compile(loss=self.loss, optimizer="adam")

    def _evaluate(self, X, Y, name):
        """Evaluate Model"""
        mse = self.model.evaluate(X, Y)
        logger.info(f"{name} mean squared error: {mse:.4f}")

    def train(self, X_train: np.ndarray, Y_train: np.ndarray):
        """Helper function to train model

        Parameters
        ----------
        x: np.ndarray
           input training data; example input: [use_id,item_id,features]
        y: np.ndarray
           input target; example movie ratings
        """

        self.build_model(X_train, Y_train)
        early_stop = EarlyStopping(
            monitor="val_loss", mode="auto", verbose=0, patience=5
        )
        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            shuffle=True,
            verbose=1,
            callbacks=[early_stop],
        ).history

        self._evaluate(X_train, Y_train, "train")
        return history


def get_recommendations(
    user_id: int,
    items: List[int],
    items_lookup: dict,
    model: Model,
    topk: int = None,
):
    """
    Helper function to predict ratings for given input user
    and return top recommended items

    Parameters
    ----------
    user_id: int
        input user id
    items: List[int]
        input items for predictions
    items_lookup: dict
        lookup items to map encodings back to unique name/id
    model: tf.keras.models.Model
        input trained matrix factorization model
    topk: int
        maximum recommended items to user
    """
    pred = model.predict([np.array([user_id] * len(items)), np.array(items)]).ravel()
    mappings = [items_lookup[i] for i in np.argsort(pred)[:topk]]
    return mappings
