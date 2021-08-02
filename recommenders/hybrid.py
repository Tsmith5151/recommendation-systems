import numpy as np
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    Concatenate,
    Reshape,
    Input,
    Embedding,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from .utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender:
    def __init__(
        self,
        unique_users: int,
        unique_items: int,
        tfidf_features: int,
        epochs: int,
        dense_units: int,
        dropout: float,
        batch_size: int,
        embedding_dim: int,
    ):
        """
        Deep Hybrid Recommender Engine:

        One of the advantage of using neural networks for recommendation
        systems is the ability to create an architecture that utilizes both
        the collaborative and content based filtering approaches. This class
        exploits using explicit data to include side features for user/items.
        Ideally this type of approach could help address the cold start problem
        or refined ranking given a subset of items filtered from a candidate
        generator model (e.g. see retrieval.py)

        Parameters
        ----------
        unique_users: np.ndarray
            input array of unique users for creating embedding layer
        unique_items: np.ndarray
            input array of unique items for creating embedding layer
        tfidf: np.ndarray
            input array of tfidf features
        epochs: int
            number of epochs to train the model. An epoch is an iteration over the
            entire x and y data provided.
        dense_units:int
            dimensionality of the output space for hidden layer
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
        self.tfidf_features = tfidf_features
        self.dropout = dropout
        self.units = dense_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.loss = "mse"

    def __repr__(self):
        return """ Deep Hybrid Recommendation Engine """

    def build_model(self, x, y):
        """Build Hybrid Model
        This helper function for generating the model can be
        extended to incorporate additional hidden features.

        Parameters
        ----------
        x: np.ndarray
           input training data; example input: [use_id,item_id,features]
        y: np.ndarray
           input target
        """

        # TFIDF Feature Vector - Item Feature
        self.tfidf_input = Input(shape=(self.tfidf_features.shape[1]), name="tfidf")
        self.tfidf_vector = Dense(64, activation="relu")(self.tfidf_input)

        # Meta Item Feature
        self.item_meta_input = Input(shape=[1], name="item_meta_feature")
        self.item_meta_vector = Dense(64, activation="relu")(self.item_meta_input)

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
        self.user_vector = Dense(64, activation="relu")(self.user_vector)

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
        self.item_vector = Dense(64, activation="relu")(self.item_vector)

        # concatentate items vector
        concat_items_vector = Concatenate()(
            [
                self.item_vector,
                self.tfidf_vector,
                self.item_meta_vector,
            ]
        )

        # concatenate user/items vector followed by dense layer(s)
        user_items_vector = Concatenate(name="user_items")(
            [self.user_vector, concat_items_vector]
        )
        layer_1 = Dense(self.units, kernel_regularizer="l2", activation="relu")(
            user_items_vector
        )
        dropout = Dropout(self.dropout)(layer_1)
        layer_2 = Dense(self.units, kernel_regularizer="l2", activation="relu")(dropout)
        self.output = Dense(1)(layer_2)

        self.model = Model(
            inputs=[
                self.user_id_input,
                self.item_id_input,
                self.tfidf_input,
                self.item_meta_input,
            ],
            outputs=self.output,
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
                input training data; example input: [use_id,item_id,features, *args]
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
            validation_split=0.1,
            shuffle=True,
            verbose=1,
            callbacks=[early_stop],
        ).history

        self._evaluate(X_train, Y_train, "train")
        return history
