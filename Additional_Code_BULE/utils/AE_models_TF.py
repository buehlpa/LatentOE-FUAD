"""
contains differnet models for anomaly detection Autoencoder
for tensorflow 

# model inputs set for FMNIST_feat features with 2048 dimentional vectors
"""

#environment needed:
#activate conda env AutencoderTF env tf '2.10.0' , python 3.9.16

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


## autoencoder:

class Autoencoder_features_simple_1(Model):
  def __init__(self, latent_dim):
    super(Autoencoder_features_simple_1, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(2048, activation='sigmoid'),
      # layers.Reshape((,2048))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class Autoencoder_features_simple_2():
    # simple autoencoder 2 form https://towardsdatascience.com/anomaly-detection-using-autoencoders-5b032178a1ea
    def __init__(self,):
        super(Autoencoder_features_simple_2, self).__init__()
        self.hidden_dim_1=1024
        self.hidden_dim_2=512
        self.input_dim=2048
        self.encoding_dim=64
        self.learning_rate=1e-7

    def model(self,):    
        input_layer = tf.keras.layers.Input(shape=(self.input_dim, ))
        #Encoder
        encoder = tf.keras.layers.Dense(self.encoding_dim, activation="tanh",activity_regularizer=tf.keras.regularizers.l2(self.learning_rate))(input_layer)
        encoder=tf.keras.layers.Dropout(0.2)(encoder)
        encoder = tf.keras.layers.Dense(self.hidden_dim_1, activation='relu')(encoder)
        encoder = tf.keras.layers.Dense(self.hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)
        # Decoder
        decoder = tf.keras.layers.Dense(self.hidden_dim_1, activation='relu')(encoder)
        decoder=tf.keras.layers.Dropout(0.2)(decoder)
        decoder = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(decoder)
        decoder = tf.keras.layers.Dense(self.input_dim, activation='tanh')(decoder)
        #Autoencoder
        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        return(autoencoder)

class Autoencoder_features_simple_3():
    # simple autoencoder 2 form https://towardsdatascience.com/anomaly-detection-using-autoencoders-5b032178a1ea
    
    def __init__(self,):
        #super(Autoencoder_features_simple_3, self).__init__()
        self.hidden_dim_1=1024
        self.hidden_dim_2=512
        self.input_dim=2048
        self.encoding_dim=64
        self.learning_rate=1e-7
        self.latentdim=64

    def model(self,):    
            input_layer = tf.keras.layers.Input(shape=(self.input_dim, ))
            #Encoder
            encoder = tf.keras.layers.Dense(self.input_dim, activation="tanh",activity_regularizer=tf.keras.regularizers.l2(self.learning_rate))(input_layer)
            encoder=tf.keras.layers.Dropout(0.2)(encoder)
            encoder = tf.keras.layers.Dense(self.hidden_dim_1, activation='relu')(encoder)
            encoder = tf.keras.layers.Dense(self.hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)

            #latentdim
            latent = tf.keras.layers.Dense(self.latentdim, activation='relu')(encoder)

            # Decoder
            decoder = tf.keras.layers.Dense(self.hidden_dim_2, activation='relu')(latent)
            decoder=tf.keras.layers.Dropout(0.2)(decoder)
            decoder = tf.keras.layers.Dense(self.hidden_dim_1, activation='relu')(decoder)
            decoder = tf.keras.layers.Dense(self.input_dim, activation='tanh')(decoder)
            #Autoencoder
            autoencoder3 = tf.keras.Model(inputs=input_layer, outputs=decoder)

            autoencoder3.compile(optimizer='adam', loss=losses.MeanSquaredError())
            return(autoencoder3)


