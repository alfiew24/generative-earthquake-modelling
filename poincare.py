import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To filter TensorFlow warnings for a cleaner log
from warnings import filterwarnings; filterwarnings('ignore') # To filter warnings for a cleaner log
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize



EPS1 = 1e-5
EPS2 = 1e-8
MAX_TANH_ARG = 15.0

def atanh(x):
    """
    This function takes some positive scalar value x and computes arctanh(x), ensuring bounds for x.
    """
    return tf.atanh(tf.clip_by_value(x, -1.0 + EPS2, 1.0 - EPS2))

def tanh(x):
    """
    This function takes some positive scalar value x and computes tanh(x), ensuring bounds for x.
    """
    return tf.tanh(tf.clip_by_value(x, -MAX_TANH_ARG, MAX_TANH_ARG))

def dot(x, y):
    """
    This function computes the Euclidean dot product between two vectors, used extensively.
    """
    return tf.reduce_sum(x * y, axis=-1, keepdims=True)

def lambda_(z, c=1.0):
    """
    This function computes the lambda coefficient for a given vector and curvature.
    """
    return 2.0 / (1.0 - c*dot(z, z))

def mob_add(u, v, c=1.0):
    """
    This function computes the resulting vector of the mobius addition between two vectors.

    Args:
        u (tf.tensor): The first hyperbolic vector to use in mobius addition.
        v (tf.tensor): The second hyperbolic vector to use in mobius addition.
        c (float): The negative curvature of the Poincaré ball (ie curvature = -c).

    Returns:
        tf.tensor: The resulting vector, clipped to the Poincaré ball.
    """
    u_dot_v = dot(u, v)
    u_dot_u = dot(u, u)
    v_dot_v = dot(v, v)
    numer = (1.0 + 2.0*c*u_dot_v + c*v_dot_v)*u + (1.0 - c*u_dot_u)*v
    denom = 1.0 + 2.0*c*u_dot_v + tf.square(c)*u_dot_u*v_dot_v + EPS2
    result = tf.clip_by_norm(numer/denom, clip_norm=(1.0 - EPS1) / tf.sqrt(c), axes=[1]) # Clipping to open ball
    return result

def exp_map(x, v, c=1.0):
    """
    This function computes the exponential map in the Poincaré space of v from some point x, exp_x(v).

    Args:
        x (tf.tensor): The base vector to exponentiate from.
        v (tf.tensor): The vector to exponentiate.
        c (float): The negative curvature of the Poincaré ball (ie curvature = -c).

    Returns:
        tf.tensor: The resulting vector.
    """
    norm_v = tf.sqrt(dot(v, v))
    sqrt_c = tf.sqrt(c)
    second_term = tanh(sqrt_c * lambda_(x, c) * norm_v / 2) * v / (sqrt_c * norm_v + EPS1)
    return mob_add(x, second_term, c)

def exp_map_0(v, c=1.0):
    """
    This function computes the exponential map in the Poincaré space of v from zero.
    """
    norm_v = tf.sqrt(dot(v, v))
    sqrt_c = tf.sqrt(c)
    return tanh(sqrt_c * norm_v) * v / (sqrt_c * norm_v + EPS1)

def log_map(x, v, c=1.0):
    """
    This function computes the natural logarithm map in the Poincaré space of v from some point x, log_x(v).

    Args:
        x (tf.tensor): The base vector.
        v (tf.tensor): The vector to take the natural log of.
        c (float): The negative curvature of the Poincaré ball (ie curvature = -c).

    Returns:
        tf.tensor: The resulting vector.
    """
    diff = mob_add(-x, v, c)
    norm_diff = tf.sqrt(dot(diff, diff))
    sqrt_c = tf.sqrt(c)
    return 2.0 / (sqrt_c * lambda_(x, c)) * atanh(sqrt_c * norm_diff) * diff / (norm_diff + EPS1)

def d_p(x, y, c=1.0):
    """
    This function compputes the distance between two vectors in the Poincaré ball space.
    """
    diff2 = dot(x - y, x - y)
    denom = (1 - c * dot(x, x)) * (1 - c * dot(y, y)) + EPS1
    return 1.0 / tf.sqrt(c) * tf.acosh(1.0 + 2.0 * c * diff2 / denom + EPS1)

def pairwise_poincare_distances(x, c=1.0):
    """
    This function computes pairwise Poincaré distances between all rows of x.

    Args:
        x (tf.tensor): Vector of samples to compare to each other.
        c (float): Negative curvature of the Poincaré ball space.

    Returns: 
        tf.tensor: Matrix of shape (n, n) with the distances between each pair of rows.
    """
    diff_norm_sq = tf.reduce_sum(tf.square(x[:, None, :] - x[None, :, :]), axis=-1)  # shape (n, n)
    x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)  # shape (n, 1)
    # (1 - c||x||^2)(1 - c||y||^2), matrix multiplication to compare all rows to each other
    denom = tf.matmul(1.0 - c*x_norm_sq, tf.transpose(1.0 - c*x_norm_sq)) + EPS1  # shape (n, n)
    return 1.0 / tf.sqrt(c) * tf.acosh(1.0 + 2.0 * c * diff_norm_sq / denom + EPS1)

def log_wrapped_normal_pdf(z, mu, ln_sigma2, latent_dim, c=1.0):
    """
    This function computes the probability density of z for a log wrapped normal with mean mu and log variance ln_sigma2.

    Args:
        z (tf.tensor): Points to compute the probability density of observing under the wrapped normal.
        mu (tf.tensor): The mean of the wrapped normal.
        ln_sigma2 (tf.tensor): The log variance of the wrapped normal.
        latent_dim (int): The number of dimensions in the space.
        c (float): Negative curvature of the Poincaré ball space.

    Returns:
        tf.tensor: The probability of observing each point in z.
    """
    sigma2 = tf.exp(ln_sigma2)
    z_ = lambda_(mu, c) * log_map(mu, z, c)
    sqrt_c_d = tf.clip_by_value(tf.sqrt(c) * d_p(mu, z, c), -20.0, 20.0)
    norm_pdf = -0.5 * tf.reduce_sum(tf.math.log(2.0 * np.pi * sigma2) + tf.square(z_)/sigma2, axis=1)
    inv_jac = (latent_dim - 1.0) * tf.math.log((sqrt_c_d + EPS2) / (tf.sinh(sqrt_c_d) + EPS2))
    return norm_pdf + inv_jac

def poincare_mean_var(x, c=1.0):
    """
    This function computes the Frechét mean and empirical variance of points in Poincaré ball geometry.
    """
    def var(mu):
        mu = exp_map_0(tf.cast([mu], tf.float32), c)[0]
        return tf.reduce_mean(d_p(tf.tile(mu[None, :], [x.shape[0], 1]), tf.cast(x, tf.float32), c)**2.0)

    frechet_mu = minimize(var, log_map(tf.zeros((x.shape[1], 1), dtype=tf.float32), tf.reduce_mean(x, axis=0, keepdims=True), c)[0]).x

    return exp_map_0(tf.cast([frechet_mu], tf.float32), c)[0], var(frechet_mu).numpy()



class GyroplaneLayer(tf.keras.layers.Layer):
    """
    This custom layer maps from the Poincaré ball back to Euclidean geometry by applying affine transformations.
    """
    def __init__(self, units, c=1.0, activation=None, **kwargs):
        super(GyroplaneLayer, self).__init__(**kwargs)
        self.units = units          # Number of output neurons
        self.c = c                  # Curvature of the Poincaré ball
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        dim = int(input_shape[-1])

        # p: centre points in the Poincaré ball
        self.p = self.add_weight(
            shape=(self.units, dim),
            initializer="random_normal",
            trainable=True,
            name="p_centres"
        )

        # a: gyroplane normal vectors
        self.a = self.add_weight(
            shape=(self.units, dim),
            initializer="random_normal",
            trainable=True,
            name="gyroplane_normals"
        )

    def call(self, inputs):
        """
        This function completes a forward pass of the layer, returning a matrix of size (inputs.shape[0], units).
        """
        sqrt_c = tf.sqrt(tf.constant(self.c, dtype=inputs.dtype))
    
        x = tf.expand_dims(inputs, 1)
        mob = mob_add(-self.p, x, self.c)
        mob_a = tf.reduce_sum(mob * self.a, axis=-1)
        norm_mob_sq = tf.reduce_sum(mob**2, axis=-1)
        norm_a = tf.norm(self.a, axis=-1)
        numerator = 2 * sqrt_c * tf.abs(mob_a)
        denominator = (1 - self.c * norm_mob_sq) * norm_a
        
        out = tf.asinh(numerator / (denominator + EPS1)) / sqrt_c

        # Optional activation
        if self.activation is not None:
            out = self.activation(out)

        return out