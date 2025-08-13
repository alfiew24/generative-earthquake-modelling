import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To filter TensorFlow warnings for a cleaner log
from warnings import filterwarnings; filterwarnings('ignore') # To filter warnings for a cleaner log
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict
from poincare import GyroplaneLayer, exp_map, exp_map_0, lambda_, log_wrapped_normal_pdf, pairwise_poincare_distances
from data import train_test_split



def build_residual_network(data_shape, latent_dim, name:str, num_blocks:int=2, proportions:list=[1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5], c:float=1.0):
    """
    This function builds the residual network architecture for the encoder or decoder of a VAE.
    """
    inputs = tf.keras.Input(shape=data_shape if name == 'Encoder' else (latent_dim,))

    if name == 'Decoder': 
        proportions.reverse()
        x = GyroplaneLayer(int(data_shape[0]*proportions[0]), c, activation='relu')(inputs)
    else:
        x = tf.keras.layers.Dense(int(data_shape[0]*proportions[0]), activation='relu')(inputs)
    
    # Adding required residual blocks
    for n in range(min(num_blocks, len(proportions)-1)):
        shortcut = x
        x = tf.keras.layers.Dense(int(data_shape[0]*proportions[n]))(x)
        x = tf.keras.layers.Dense(int(data_shape[0]*proportions[n]))(x)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(int(data_shape[0]*proportions[n+1]), activation='relu')(x)

    # Final layers
    for n in range(num_blocks+1, len(proportions)):
        x = tf.keras.layers.Dense(int(data_shape[0]*proportions[n]), activation='relu')(x)
    
    if name == 'Encoder':
        outputs = tf.keras.layers.Dense(latent_dim * 2)(x)
    else:
        outputs = tf.keras.layers.Dense(data_shape[0])(x)

    return tf.keras.Model(inputs, outputs, name=name)



class poincare_hierarchical_vae:
    """
    This class creates an instance of the Poincaré Hierarchical Variational Auto-Encoder model (VAE).
    The model requires the feature space shape and number of latent dimensions to be defined on initialisation.
    """
    def __init__(self, data_shape:tuple, latent_dim:int=2, c:float=1.0, rec_weight=20.0, kl_weight=1.0, reg_weight=100.0, kl_hyp_weight=50.0, reg_hyp_weight=50.0, s:float=1.0, grp_var:float=0.01, hyp_var:float=0.05):
        """
        The initialisation method creates the two neural networks of the model, defines the optimisers/learning rates, and defines the global variables.

        Args:
            data_shape (tuple): The shape of the data that the model is trying to generate.
            latent_dim (int): The number of latent variables.
            c (float): The negative curvature of the Poincaré ball latent space.
            rec_weight (float): The weight of the reconstruction loss term.
            kl_weight (float): The weight of the KL-divergence loss term.
            reg_weight (float): The weight of the sample-level regularisation loss term.
            kl_hyp_weight (float): The weight of the hyperprior KL-divergence loss term.
            reg_hyp_weight (float): The weight of the event-level regularisation loss term.
            s (float): The base spreading term used in the dispersion part of the regularisation loss term.
            grp_var (float): The variance for each of the group prior distributions (constant).
            hyp_var (float): The variance used for the hyper-prior distribution.
        """
        self.c = c
        self.latent_dim = latent_dim

        self.encoder = build_residual_network(data_shape, latent_dim, 'Encoder', 1, [1.0, 0.95, 0.9, 0.85], self.c)
        self.decoder = build_residual_network(data_shape, latent_dim, 'Decoder', c=self.c)

        self.enc_optimiser = tf.keras.optimizers.Adam(1e-5)
        self.dec_optimiser = tf.keras.optimizers.Adam(1e-4)
        self.lr_grp = tf.Variable(5e-4, trainable=False, dtype=tf.float32)

        self.W = tf.constant([rec_weight, kl_weight, reg_weight, kl_hyp_weight, reg_hyp_weight], dtype=tf.float32)
        self.s = s
        self.grp_var = grp_var
        self.hyp_var = hyp_var

    def reparam(self, mu, ln_sigma2):
        """
        The reparameterisation trick using a wrapped normal distribution.
        """
        batch_size = tf.cast(tf.shape(mu)[0], tf.int32)
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        z = (tf.exp(ln_sigma2) ** 0.5) * eps
        return exp_map(mu, z / lambda_(mu, self.c), self.c)
    
    def loss(self, x, eqids, reg_data, training:bool=True):
        """
        This method computes the loss given some data. It uses the encoder and decoder to generate a synthetic copy of the data.
        Returns the five loss components as a tensor.
        """
        # Computing the reconstructed data as well as pulling out the latent variables
        latent_embedding = self.encoder(x, training=training)
        mu, ln_sigma2 = tf.split(latent_embedding, num_or_size_splits=2, axis=1)
        mu = exp_map_0(mu, self.c)
        ln_sigma2 = tf.clip_by_value(ln_sigma2, -5, 10) # Clipping the values to prevent infinities
        z = self.reparam(mu, ln_sigma2)
        x_recon = self.decoder(z, training=training)
        group_means_ = exp_map_0(self.group_means, self.c)
        B = tf.cast(tf.shape(eqids)[0], tf.int32) # Getting the current batch size

        # Computing reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(x - x_recon))

        # Computing the main KL-divergence loss
        mu_g = tf.gather(group_means_, tf.cast(eqids, tf.int32))
        log_q = log_wrapped_normal_pdf(z, mu, ln_sigma2, self.latent_dim, self.c)
        log_p = log_wrapped_normal_pdf(z, mu_g, tf.math.log(self.grp_var), self.latent_dim, self.c)
        kl_loss = tf.reduce_mean(log_q - log_p)

        # Computing the sample-level regularisation loss
        variances = tf.tile(tf.expand_dims(tf.expand_dims(self.reg_variances, axis=0), axis=1), [B, B, 1])
        all_prods_sq = tf.square(reg_data[:, None, :] - reg_data[None, :, :]) / variances
        diff_sq = pairwise_poincare_distances(z, self.c)
        # Adhesion
        full_comp = tf.cast(tf.exp(-tf.reduce_sum(all_prods_sq, axis=2)), tf.float32) * diff_sq
        full_comp = tf.boolean_mask(full_comp, tf.equal(tf.expand_dims(eqids, axis=1), tf.expand_dims(eqids, axis=0)))
        reg_loss = tf.reduce_sum(full_comp) / tf.cast(B ** 2, tf.float32)
        # Dispersion
        full_comp2 = tf.cast(tf.exp(-diff_sq), tf.float32) * tf.reduce_sum(all_prods_sq, axis=2)
        full_comp2 = tf.boolean_mask(full_comp2, tf.equal(tf.expand_dims(eqids, axis=1), tf.expand_dims(eqids, axis=0)))
        reg_loss += tf.reduce_sum(full_comp2) / tf.cast(B ** 2, tf.float32)
        
        # Computing the hyper-prior KL-divergence loss
        mu_g_bar = tf.reduce_mean(self.group_means, axis=0)
        mu_g_sigma2 = tf.math.reduce_variance(self.group_means, axis=0)
        kl_hyp_loss = 0.5 * tf.reduce_mean(tf.math.log(self.hyp_var) - tf.math.log(mu_g_sigma2+1e-6) + mu_g_sigma2/self.hyp_var + tf.square(mu_g_bar)/self.hyp_var - 1.0)

        # Computing the event-level regularisation loss
        N = self.reg_hyp_data.shape[0]
        variances = tf.tile(tf.expand_dims(tf.expand_dims(self.reg_hyp_variances, axis=0), axis=1), [N, N, 1])
        all_prods_sq = tf.square(self.reg_hyp_data[:, None, :] - self.reg_hyp_data[None, :, :]) / variances
        diff_sq = pairwise_poincare_distances(group_means_, self.c)
        # Keeping similar means close - adhesion part
        reg_hyp_loss = tf.reduce_sum(tf.cast(tf.exp(-tf.reduce_sum(all_prods_sq, axis=2)), tf.float32) * diff_sq) / (N ** 2)
        # Spreading means out, especially different ones - dispersion part
        reg_hyp_loss += tf.reduce_sum(tf.cast(tf.exp(-diff_sq), tf.float32) * (tf.reduce_sum(all_prods_sq, axis=2) + self.s)) / (N ** 2)

        return self.W * tf.stack([recon_loss, kl_loss, reg_loss, kl_hyp_loss, reg_hyp_loss])

    @tf.function
    def train_step(self, dataset):
        """
        This method completes one training step (epoch) given some batched dataset.
        Returns the average five loss components over all batches as a tensor.
        """
        L = tf.zeros(5, dtype=tf.float32)
        batches = 0

        for batch_spectrum, batch_eqids, batch_data in dataset:
            # Computing loss
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.group_means)
                loss = self.loss(batch_spectrum, batch_eqids, batch_data)
                loss_all = tf.reduce_sum(loss)

            # Computing gradients
            enc_grad = tape.gradient(loss_all, self.encoder.trainable_variables)
            dec_grad = tape.gradient(loss_all, self.decoder.trainable_variables)
            grp_grad = tape.gradient(loss_all, [self.group_means])
            del tape

            # Applying gradients
            self.enc_optimiser.apply_gradients(zip(enc_grad, self.encoder.trainable_variables))
            self.dec_optimiser.apply_gradients(zip(dec_grad, self.decoder.trainable_variables))
            
            # Applying group effect gradients - only to the groups used for this batch of training
            group_indices, _ = tf.unique(tf.cast(batch_eqids, tf.int32))
            mean_grd_smp = tf.gather(grp_grad[0], group_indices)
            self.group_means.assign(tf.tensor_scatter_nd_sub(self.group_means, 
                                                             tf.expand_dims(group_indices, 1), self.lr_grp * mean_grd_smp))

            L += loss
            batches += 1

        B = tf.cast(batches, tf.float32)
        return L / B
    
    @tf.function
    def validation_loss(self, dataset):
        """
        This method computes the loss for a batched dataset without applying gradients. For validation.
        Returns the average five loss components over all batches as a tensor.
        """
        L = tf.zeros(5, dtype=tf.float32)
        batches = 0
        for batch_spectrum, batch_eqids, batch_data in dataset:
            L += self.loss(batch_spectrum, batch_eqids, batch_data, training=False)
            batches += 1
        
        B = tf.cast(batches, tf.float32)
        return L / B
    
    def create_batches(self, spectrum, eqids, data, batch_size=64):
        """
        This method creates a batched tensorflow dataset for the provided data, ensuring samples from the same group are kept together.
        Returns the batched tensorflow dataset.
        """
        # Group indices by eqid
        eqids = np.array(eqids)
        group_dict = defaultdict(list)
        for idx, eqid in enumerate(eqids):
            group_dict[eqid].append(idx)

        # Shuffle group order
        all_group_ids = list(group_dict.keys())
        np.random.shuffle(all_group_ids)

        # Build batches (based on number of samples, not groups)
        batches = []
        current_batch = []
        current_batch_size = 0
        for gid in all_group_ids:
            group_indices = group_dict[gid]
            if current_batch_size + len(group_indices) > batch_size and current_batch:
                batches.append(np.concatenate(current_batch))
                current_batch = []
                current_batch_size = 0
            current_batch.append(group_indices)
            current_batch_size += len(group_indices)
        if current_batch:
            batches.append(np.concatenate(current_batch))

        # Generator that yields batches
        def gen():
            for batch_indices in batches:
                yield (
                    tf.convert_to_tensor(spectrum[batch_indices], dtype=tf.float32),
                    tf.convert_to_tensor(eqids[batch_indices], dtype=tf.int32),
                    tf.convert_to_tensor(data[batch_indices], dtype=tf.float32),
                )

        output_types = (tf.float32, tf.int32, tf.float32)
        output_shapes = (tf.TensorShape([None, spectrum.shape[1]]), tf.TensorShape([None]), tf.TensorShape([None, data.shape[1]]))
        dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def train(self, response_spectrum:pd.DataFrame, data:pd.DataFrame, reg_feats:list, reg_hyp_feats:list, min_epochs:int=100, max_epochs:int=1000, batch_size:int=400):
        """
        This method completes one full training session on the given real data. Training will stop when the minimum number of epochs have been completed and
        either there was negligible change in loss over the past two epochs or the maximum number of epochs is reached.
        
        Args:
            response_spectrum (pandas dataframe): Dataframe containing response spectrum samples for the model to train with.
            data (pandas dataframe): Dataframe containing features for regularisation (EQID must be included).
            reg_feats (list): A list of features to use for sample-level regularisation (column names).
            reg_hyp_feats (list): A list of features to use for group-level regularisation (columns names).
            min_epochs (int): The number of training steps (epochs) to complete before convergence starts to be checked.
            max_epochs (int): The maximum number of training steps (epochs) to complete. If convergence is detected earlier, then the training stops.
            batch_size (int): The size of the batches, parameters are adjusted after every batch.
        
        Returns:
            numpy array: A matrix of training losses from every epoch, (num_epochs, num_loss_terms).
            numpy array: A matrix of validation losses from every epoch, (num_epochs, num_loss_terms).
        """
        assert 'EQID' in data.columns, 'EQID must be in data.'
        self.reg_variances = tf.constant(data[reg_feats].to_numpy().var(axis=0), tf.float32) # Getting variances so it's not computed per batch
        data_g = data.groupby('EQID').mean().reset_index() # Grouping by EQID for event level regularisation

        # Encoding EQIDs as indices
        data_ = data.copy()
        data_['EQID'] = data_['EQID'].apply(lambda x: np.where(data_g['EQID'] == x)[0][0])
        data_g['EQID'] = data_g['EQID'].apply(lambda x: np.where(data_g['EQID'] == x)[0][0])

        # Train-validation split, 10% for validation (reserving highest magnitude for validation)
        train_response_spectrum, val_response_spectrum, train_data, val_data = train_test_split(response_spectrum, data_, 0.1)
        train_eqids, val_eqids = train_data['EQID'], val_data['EQID']
        train_data, val_data = train_data[reg_feats], val_data[reg_feats]

        print(f'Training dataset size : {train_response_spectrum.shape}\t Validation dataset size : {val_response_spectrum.shape}')

        # Setting up group effects
        self.group_means = tf.Variable(1e-6*tf.random.normal((len(data_g), self.latent_dim)), dtype=tf.float32, trainable=True)
        self.reg_hyp_data = tf.constant(data_g[reg_hyp_feats].to_numpy(), tf.float32)
        self.reg_hyp_variances = tf.constant(data_g[reg_hyp_feats].to_numpy().var(axis=0), tf.float32)

        # Lists to store all loss values from training, embeddings and group means every 5 epochs 
        train_losses, val_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True), tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for n in range(max_epochs):
            # Batching the dataset before training
            dataset = self.create_batches(train_response_spectrum.to_numpy(), train_eqids.to_numpy(), train_data.to_numpy(), batch_size)
            train_loss = self.train_step(dataset)
            train_losses = train_losses.write(n, train_loss)

            # Validation batching and loss computation
            val_dataset = self.create_batches(val_response_spectrum.to_numpy(), val_eqids.to_numpy(), val_data.to_numpy(), batch_size)
            val_loss = self.validation_loss(val_dataset)
            val_losses = val_losses.write(n, val_loss)

            #print(f'{n}: Train Loss : {tf.reduce_sum(train_loss):.2f}\tValidation Loss : {tf.reduce_sum(val_loss):.2f}', end='\r')
            print(n, np.array2string(train_loss.numpy(), formatter={'float_kind':lambda x: "%12.2f" % x}), end='\r')

            # Early breaking checks
            if tf.reduce_any(tf.math.is_nan(train_loss)) or train_loss[1] < 0.0: break
            if n > min_epochs:
                ts = train_losses.stack()
                vs = val_losses.stack()
                if tf.reduce_max(tf.reduce_mean(ts[-11:-1] - ts[-10:], axis=0) / ts[-11]) < 1e-3 or tf.reduce_max(tf.reduce_mean(vs[-11:-1] - vs[-10:], axis=0) / vs[-11]) < 1e-3:
                    break

        print('')
        return train_losses.stack().numpy(), val_losses.stack().numpy()