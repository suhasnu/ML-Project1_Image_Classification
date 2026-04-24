"""
Visualize latent space rep --> make it square!
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
import matplotlib.pyplot as plt
import sys 
from sklearn.preprocessing import MinMaxScaler

latent_dim = 25
EPOCHS = 100 ;


"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

"""
## Define the VAE as a `Model` with a custom `train_step`
"""
class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ## Build the decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # ---
        #decoder_outputs_mean = layers.Reshape((784,))(layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x))
        #decoder_outputs_logvar = layers.Reshape((784,))(layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x))
        #sampled = Sampling()([decoder_outputs_mean, decoder_outputs_logvar]) ;
        #decoder_outputs = layers.Reshape((28,28,1))(sampled) ;
        # ---
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        # ---
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.build(input_shape = (latent_dim,)) ; 
        self.decoder.summary()

        ## Build the encoder
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.build(input_shape = (28,28,1)) ; 
        self.encoder.summary() ;

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #reconstruction_loss = ops.mean(
            #    ops.sum(
            #        keras.losses.binary_crossentropy(data, reconstruction),
            #        axis=(1, 2),
            #    )
            #)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum((data-reconstruction)**2, axis=(1,2,3))) ;
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }


if __name__ == "__main__":
  (x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data() ;
  mnist_digits_train = np.reshape(x_train, (-1, 28,28,1)).astype("float32") / 255.
  mnist_digits_test = np.reshape(x_test, (-1, 28,28,1)).astype("float32") / 255.
  mnist_digits_train_0 = mnist_digits_train[t_train == 0] ;
  mnist_digits_test_0 = mnist_digits_test[t_test == 0] ;
  mnist_digits_train_1 = mnist_digits_train[t_train == 1] ;
  mnist_digits_test_1 = mnist_digits_test[t_test == 1] ;

  vae = VAE()
  vae.compile(optimizer=keras.optimizers.Adam())
  if sys.argv[1] == "train":
    vae.fit(mnist_digits_train_0, epochs=EPOCHS, batch_size=128)
    vae.encoder.save_weights("vae_enc.weights.h5") ;
    vae.decoder.save_weights("vae_dec.weights.h5") ;
  else:
    vae.decoder.load_weights("vae_dec.weights.h5") ;
    vae.encoder.load_weights("vae_enc.weights.h5") ;

  # -------- you code here!
  
  def plot_latent_space(vae, digits, label, n=10):
    z, _, _ = vae.encoder(digits[:100])
    figure = z.numpy().reshape(n, n, 5, 5).transpose(0, 2, 1, 3).reshape(n*5, n*5)
    plt.figure(figsize=(8,8))
    plt.imshow(figure)
    plt.title(label)
    plt.savefig("latent_space.png")
    plt.show()
    
  def plot_generated_samples(vae, n=10):
    generated = vae.decoder(np.random.normal(size=(n*n, latent_dim)).astype("float32")).numpy()
    figure = generated.reshape(n, n, 28, 28).transpose(0, 2, 1, 3).reshape(n*28, n*28)
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.title("Generated Samples")
    plt.savefig("Samples.png")
    plt.show()
    
  plot_latent_space(vae, mnist_digits_test_0, "Latent Space - Class 0")
  plot_latent_space(vae, mnist_digits_test_1, "Latent Space - Class 1")
  plot_generated_samples(vae)
  
    
    
        
  

