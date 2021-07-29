# Matrix Factorization using Keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error


class MatrixFactorization:
    def __init__(
        self,
        n_users,
        n_items,
        epochs: int,
        learning_rate: float,
        dropout: float,
        batch_size: int,
        embedding_dim: int,
        pred_type: str = None,
    ):

        self.n_users = n_users
        self.n_items = n_items
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.loss = "mse"
        self.pred_type = pred_type

    def __repr__(self):
        return """ Matrix Factorization Model """

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)

    def user_embedding(self):
        """User Embeddings"""
        self.user_id_input = tf.keras.layers.Input(shape=[1], name="user")
        self.user_embedding = tf.keras.layers.Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.n_users,
            input_length=1,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name="user_embedding",
        )(self.user_id_input)
        self.user_vector = tf.keras.layers.Reshape([self.embedding_dim])(
            self.user_embedding
        )

    def item_embedding(self):
        """Item Embeddings"""
        self.item_id_input = tf.keras.layers.Input(shape=[1], name="item")
        self.item_embedding = tf.keras.layers.Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.n_items,
            input_length=1,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name="item_embedding",
        )(self.item_id_input)
        self.item_vector = tf.keras.layers.Reshape([self.embedding_dim])(
            self.item_embedding
        )

    def prediction(self):
        """Function to compute the dot product of te user/item vector"""

        if self.pred_type == "dot":
            prediction = tf.keras.layers.Dot(1, normalize=False)(
                [self.user_vector, self.item_vector]
            )
        else:
            # dense hidden layers
            concat_vectors = tf.keras.layers.Concatenate()(
                [self.user_vector, self.item_vector]
            )
            layer_1 = tf.keras.layers.Dense(
                128, kernel_regularizer="l2", activation="relu"
            )(concat_vectors)
            dropout = tf.keras.layers.Dropout(self.dropout)(layer_1)
            layer_2 = tf.keras.layers.Dense(
                128, kernel_regularizer="l2", activation="relu"
            )(dropout)
            prediction = tf.keras.layers.Dense(1)(layer_2)
        return prediction

    def build_model(self, x, y):
        """Build Matrix Factorization Model"""
        self.user_embedding()
        self.item_embedding()
        self.model = tf.keras.models.Model(
            inputs=[self.user_id_input, self.item_id_input], outputs=self.prediction()
        )
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, x: np.ndarray, y: np.ndarray):
        """Model Training"""

        self.build_model(x, y)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="auto", verbose=0, patience=5
        )
        history = self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
            callbacks=[early_stop],
        ).history
        return history

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        """Evaluate Model Performance"""
        y_pred = self.model.predict([x, y])
        rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y))
        print(f"Root Mean Squared Error: {rmse:.2f}")
