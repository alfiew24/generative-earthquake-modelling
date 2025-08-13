import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To filter TensorFlow warnings for a cleaner log
from warnings import filterwarnings; filterwarnings('ignore') # To filter warnings for a cleaner log
import tensorflow as tf
import numpy as np
from poincare import GyroplaneLayer, exp_map_0, poincare_mean_var



class two_stage_nn:
    def __init__(self, latent_dim:int=2, num_event_level_feats:int=2, num_sample_level_feats:int=2, poincare:bool=True, c:float=1.0):
        self.poincare = poincare
        self.c = c
        
        self.event_mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(latent_dim,)),
            GyroplaneLayer(50, self.c, activation='relu') if poincare else tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(40),
            tf.keras.layers.Dense(30),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(15),
            tf.keras.layers.Dense(num_event_level_feats)
        ])
        self.sample_mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(num_event_level_feats+latent_dim,)),
            GyroplaneLayer(50, self.c, activation='relu') if poincare else tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(40),
            tf.keras.layers.Dense(30),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(15),
            tf.keras.layers.Dense(num_sample_level_feats)
        ])

        self.event_optimiser = tf.keras.optimizers.Adam(1e-3)
        self.sample_optimiser = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step_stage1(self, dataset):
        L = 0.0
        B = 0

        for batch_centroids, batch_event_feats in dataset:
            # Computing loss
            with tf.GradientTape() as tape:
                y_pred = self.event_mlp(batch_centroids, training=True)
                loss = tf.reduce_mean(tf.square(y_pred - batch_event_feats))

            # Computing gradients
            grad = tape.gradient(loss, self.event_mlp.trainable_variables)
            self.event_optimiser.apply_gradients(zip(grad, self.event_mlp.trainable_variables))
            
            L += loss
            B += 1

        return L / tf.cast(B, tf.float32)
    
    @tf.function
    def train_step_stage2(self, dataset):
        L = 0.0
        B = 0

        for batch_samples, batch_sample_feats in dataset:
            # Computing loss
            with tf.GradientTape() as tape:
                y_pred = self.sample_mlp(batch_samples, training=True)
                loss = tf.reduce_mean(tf.square(y_pred - batch_sample_feats))

            # Computing gradients
            grad = tape.gradient(loss, self.sample_mlp.trainable_variables)
            self.sample_optimiser.apply_gradients(zip(grad, self.sample_mlp.trainable_variables))
            
            L += loss
            B += 1

        return L / tf.cast(B, tf.float32)
    
    @tf.function
    def create_batches_stage1(self, centroids, event_feats, batch_size=400):
        """
        This method creates a batched tensorflow dataset for the provided data.
        Returns the batched tensorflow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(centroids, tf.float32), tf.cast(event_feats, tf.float32)))
        return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
    
    @tf.function
    def create_batches_stage2(self, z_event_feats, sample_feats, batch_size=400):
        """
        This method creates a batched tensorflow dataset for the provided data.
        Returns the batched tensorflow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(z_event_feats, tf.float32), tf.cast(sample_feats, tf.float32)))
        return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

    def train(self, z, data, event_feats_ls, sample_feats_ls, max_epochs_stage1:int=100, max_epochs_stage2:int=100):
        z = np.array(z)
        data_ = data.copy()

        # Normalising features
        self.event_feats_mu = data[event_feats_ls].to_numpy().mean(axis=0)
        self.event_feats_var = data[event_feats_ls].to_numpy().var(axis=0)
        self.sample_feats_mu = data[sample_feats_ls].to_numpy().mean(axis=0)
        self.sample_feats_var = data[sample_feats_ls].to_numpy().var(axis=0)
        data_.loc[:, event_feats_ls] = (data[event_feats_ls] - self.event_feats_mu) / self.event_feats_var**0.5
        data_.loc[:, sample_feats_ls] = (data[sample_feats_ls] - self.sample_feats_mu) / self.sample_feats_var**0.5
        event_centroids = []
        data_g = data_.groupby('EQID').mean().reset_index()

        if self.poincare:
            for id in data_g['EQID'].values:
                z_ = z[data_['EQID'] == id]
                mean, _ = poincare_mean_var(z_)
                event_centroids.append(mean)
                data_.loc[data_['EQID'] == id, [f'C{d}' for d in range(z_.shape[1])]] = mean

        else:
            for id in data_g['EQID'].values:
                z_ = z[data_['EQID'] == id]
                mean = tf.reduce_mean(z_, axis=0)
                event_centroids.append(mean)

        centroids = tf.stack(event_centroids)
        event_feats = data_g[event_feats_ls].to_numpy()
        dataset_stage1 = self.create_batches_stage1(centroids, event_feats)

        if self.poincare:
            data_event_feats = tf.constant(data_[event_feats_ls].to_numpy(), tf.float32)
            z_event_feats = tf.concat([z, *[exp_map_0(data_event_feats[:, f:f+1]) for f in range(len(event_feats_ls))]], axis=1)
            #data_event_feats = tf.constant(data_[[f'C{d}' for d in range(z.shape[1])]].to_numpy(), tf.float32)
            #z_event_feats = tf.concat([z, *[data_event_feats[:, f:f+1] for f in range(len(event_feats_ls))]], axis=1)
        else:
            event_data_npy = data_[event_feats_ls].to_numpy()
            z_event_feats = tf.concat([z, event_data_npy], axis=1)
        sample_feats = data_[sample_feats_ls].to_numpy()
        dataset_stage2 = self.create_batches_stage2(z_event_feats, sample_feats)

        print('Training - Stage 1')
        for n in range(max_epochs_stage1):
            loss_stage1 = self.train_step_stage1(dataset_stage1)
            print(f'{n}. Loss = {loss_stage1:.4f}', end='\r')

        print('\nTraining - Stage 2')
        for n in range(max_epochs_stage2):
            loss_stage2 = self.train_step_stage2(dataset_stage2)
            print(f'{n}. Loss = {loss_stage2:.4f}', end='\r')

        print('')
        return loss_stage1, loss_stage2
    
    def predict(self, z, event_centroids):
        event_feats = self.event_mlp(event_centroids, training=False)

        if self.poincare:
            z_event_feats = tf.concat([z, *[exp_map_0(event_feats[:, f:f+1], self.c) for f in range(event_feats.shape[1])]], axis=1)
        else:
            z_event_feats = tf.concat([z, (event_feats - self.event_feats_mu) / self.event_feats_var**0.5], axis=1)
        
        sample_feats = self.sample_mlp(z_event_feats, training=False)

        return tf.concat([event_feats*self.event_feats_var**0.5 + self.event_feats_mu, 
                         sample_feats*self.sample_feats_var**0.5 + self.sample_feats_mu], axis=1).numpy()
    
    def evaluate(self, z, data, event_feats_ls, sample_feats_ls):
        samples = []
        event_centroids = []
        true_data = []
        data_g = data.groupby('EQID').mean().reset_index()

        if self.poincare:
            for id in data_g['EQID'].values:
                z_ = z[data['EQID'] == id]
                samples.append(z_)
                mean, _ = poincare_mean_var(z_)
                event_centroids.append(tf.tile(mean[None, :], [z_.shape[0], 1]))
                true_data.append(data.loc[data['EQID'] == id, [*event_feats_ls, *sample_feats_ls]].to_numpy())

        else:
            for id in data_g['EQID'].values:
                z_ = z[data['EQID'] == id]
                samples.append(z_)
                mean = tf.reduce_mean(z_, axis=0)
                event_centroids.extend([mean]*z_.shape[0])
                true_data.append(data.loc[data['EQID'] == id, [*event_feats_ls, *sample_feats_ls]].to_numpy())

        samples = tf.constant(np.concatenate(samples, axis=0), tf.float32)
        event_centroids = tf.concat(event_centroids, axis=0)
        true_data = tf.constant(np.concatenate(true_data, axis=0), tf.float32)

        pred_data = self.predict(samples, event_centroids)

        return samples, event_centroids, true_data, pred_data