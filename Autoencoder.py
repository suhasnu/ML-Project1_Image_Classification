import tensorflow as tf ;
import numpy as np ;
import sys ;
import matplotlib.pyplot as plt ;
import math ;

LATENT_DIM = 25 ;
EPOCHS=10 ;

# subclassing keras functional model
class AE(tf.keras.Model):
  def __init__(self, **kwargs):
	# mandatory
    tf.keras.Model.__init__(self, **kwargs) ;

    # define encoder
    inp = tf.keras.Input(shape=(784,)) ;
    x = tf.keras.layers.Dense(50)(inp) ;
    out = tf.keras.layers.Dense(LATENT_DIM)(x) ;
    self.encoder = tf.keras.Model(inp, out) ;
    self.encoder.build(input_shape=(784,)) ;

    # define decoder
    inp = tf.keras.Input(shape=(LATENT_DIM,)) ;
    x = tf.keras.layers.Dense(50)(inp) ;
    out = tf.keras.layers.Dense(784)(x) ;
    self.decoder = tf.keras.Model(inp, out) ;
    self.decoder.build(input_shape=(784,)) ;

    self.reconstruction_loss_tracker = tf.keras.metrics.Mean() ;

  @property
  def metrics(self):
        return [
            self.reconstruction_loss_tracker 
        ]

  def train_step(self, d, **kwargs):
    with tf.GradientTape() as g:
      latent = self.encoder(d) ;
      reconstruction = self.decoder(latent) ;
      recloss = tf.reduce_mean((d-reconstruction)**2) ;
    # self.trainable_variables gets defined by compile()
    grads = g.gradient(recloss, self.trainable_weights) ;
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights) ) ;
    self.reconstruction_loss_tracker.update_state(recloss) ;
    # train_step should return a dictionary of metrics, displayed during fit()
    return {"rec_loss":self.reconstruction_loss_tracker.result() } ;

if __name__=="__main__":
  (x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()
  scalar_labels_train = t_train ;
  scalar_labels_test = t_test ;
  train_digits = np.reshape(x_train, (-1,784)).astype("float32") / 255. ;
  test_digits = np.reshape(x_test, (-1,784)).astype("float32") / 255. ;
  train_digits_0 = train_digits[scalar_labels_train==0] ;
  print(train_digits_0.shape) ;
  test_digits0 = test_digits[scalar_labels_test==0] ;
  test_digits1 = test_digits[scalar_labels_test==1] ;

  ae = AE() ;
  ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly = False) ;
  if sys.argv[1] == "train":
    ae.fit(train_digits_0, epochs = EPOCHS, batch_size = 100) ;
    ae.encoder.save_weights("ae1.weights.h5") ;
    ae.decoder.save_weights("ae2.weights.h5") ;
  else:
    ae.encoder.load_weights("ae1.weights.h5") ;
    ae.decoder.load_weights("ae2.weights.h5") ;
  
  # ----- training is done! -------------
  
  mse_0 = np.mean((test_digits0[:100] - ae.decoder(ae.encoder(test_digits0[:100])).numpy())**2, axis=1)
  mse_1 = np.mean((test_digits1[:100] - ae.decoder(ae.encoder(test_digits1[:100])).numpy())**2, axis=1)
  
  
  thresholds = np.linspace(0, max(mse_0.max(), mse_1.max()), 100)

  plt.scatter([np.mean(mse_0 <= t) * 100 for t in thresholds],
              [np.mean(mse_1 >  t) * 100 for t in thresholds], s=10)
  plt.xlabel("inlier retention %")
  plt.ylabel("outlier rejection %")
  plt.savefig("fig1.png")
  plt.show()