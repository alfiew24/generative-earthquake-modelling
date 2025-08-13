import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To filter TensorFlow warnings for a cleaner log
from warnings import filterwarnings; filterwarnings('ignore') # To filter warnings for a cleaner log
import tensorflow as tf
import numpy as np
from poincare import exp_map_0, d_p



class ffnn:
    def __init__(self, input_dims, latent_dims):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(input_dims,)),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(55, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(45, activation='relu'),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(35, activation='relu'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(latent_dims)
        ])
        self.optimiser = tf.keras.optimizers.Adam(9e-4)

    @tf.function
    def train_step(self, dataset):
        L, B = 0.0, 0.0
        for batch_X, batch_z in dataset:
            with tf.GradientTape(persistent=True) as tape:
                pred_z = self.model(batch_X, training=True)
                pred_z = exp_map_0(pred_z)
                loss = tf.reduce_mean(d_p(batch_z, pred_z))
                #loss = tf.reduce_mean(tf.square(tf.math.log(batch_z+1.0) - tf.math.log(pred_z+1.0)))

            grad = tape.gradient(loss, self.model.trainable_variables)
            del tape
            self.optimiser.apply_gradients(zip(grad, self.model.trainable_variables))

            L += loss
            B += 1.0

        return L / B

    @tf.function
    def create_batches(self, X, z, batch_size=100):
        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X, tf.float32), tf.cast(z, tf.float32)))
        return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    def train(self, X, z, batch_size, max_epochs):
        X_ = np.log(X+1e-10)
        self.mu, self.var = X_.mean(axis=0), X_.var(axis=0)
        X_ = (X_ - self.mu) / self.var**0.5

        losses = []
        for n in range(max_epochs):
            dataset = self.create_batches(X_, z, batch_size)
            loss = self.train_step(dataset)
            print(f'{n}: {loss:.4f}', end='\r')
            losses.append(loss)

        print('')
        return losses

    def predict(self, X):
        X_ = (np.log(X+1e-10) - self.mu) / self.var**0.5
        z = self.model(X_, training=False)
        return exp_map_0(z)