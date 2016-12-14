import matplotlib as mpl

mpl.use('Agg')
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import random
import math
import numpy as np
import os.path
import logging
import logging.config
import pickle
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, merge, Merge, Activation, BatchNormalization, RepeatVector, \
    TimeDistributed, Dropout, Reshape, Permute, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.recurrent import LSTM
from keras.regularizers import l2, l1
from sklearn.utils import shuffle
from keras.optimizers import SGD, Adam
from tqdm import tqdm
import theano.tensor as T
import theano
import pprint
import os
from repeat_vector_axis import RepeatVectorAxis
from show_samples import show_samples
from show_autoencoding import show_autoencoding
from mnist_data import mnist_data


def apply_layers(x, layers):
    h = x
    for layer in layers:
        h = layer(h)
    return h


if K.backend() == "tensorflow":
    import tensorflow as tf


    def cumsum(z):
        shape = tf.shape(z)
        ia = tf.zeros(shape[1:])
        ib = tf.zeros(shape[2:])
        ic = tf.zeros(shape[3:])
        return tf.scan(
            lambda _, a: tf.scan(lambda _, b: tf.scan(lambda c, d: c + d, b, initializer=ic), a, initializer=ib), z,
            initializer=ia)


    def cumsum_single(z):
        shape = tf.shape(z)
        ia = tf.zeros(shape[1:])
        ic = tf.zeros(shape[2:])
        return tf.scan(lambda _, a: tf.scan(lambda c, d: c + d, a, initializer=ic), z, initializer=ia)
else:
    import theano.tensor as T


    def cumsum(z):
        return T.cumsum(z, axis=2)


    def cumsum_single(z):
        return T.cumsum(z, axis=1)


def model_set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        model.trainable = trainable
        if hasattr(layer, 'layer') and layer.layer is not None:
            layer.layer.trainable = trainable


def parse_metrics(metrics, model):
    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]
    return {name: metrics[i] for i, name in enumerate(model.metrics_names)}


def n_choice(x, n):
    return x[np.random.choice(x.shape[0], size=n, replace=False), :]


def leaky_relu(x):
    return K.relu(x, alpha=1.0 / 5.5)


class BiGRAN(object):
    @classmethod
    def mnist_model(cls,
                    latent_dim=25, input_dim=28 * 28, batch_size=64,
                    t=4,
                    lr_d=3e-4,
                    lr_g=1e-4,
                    minibatch_size=4,
                    decay_d = 1e-4,
                    decay_g = 4e-4
                    ):
        model = cls(input_dim=input_dim,
                    latent_dim=latent_dim,
                    batch_size=batch_size,
                    lr_d=lr_d, lr_g=lr_g,
                    t=t, minibatch_size=minibatch_size, decay_d=decay_d, decay_g=decay_g
                    )
        return model

    def set_trainable(self, is_generator):
        model_set_trainable(self.model_encoder, is_generator)
        model_set_trainable(self.model_decoder, is_generator)
        model_set_trainable(self.model_discriminator, not is_generator)

    def __init__(self, input_dim, latent_dim, t, batch_size, lr_d, lr_g, minibatch_size, decay_d, decay_g):
        self.minibatch_size = minibatch_size
        self.opt_discriminator = Adam(lr=lr_d, decay=decay_d)
        self.opt_generator = Adam(lr=lr_g, decay=decay_g)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.t = t
        self.create_models()
        self.log = None

    def encode(self, x, epsilons):
        return self.model_encoder_single.predict([x, epsilons], verbose=0, batch_size=self.batch_size)

    def decode(self, zs):
        return self.model_decoder_single.predict(zs, verbose=0, batch_size=self.batch_size)

    def autoencode(self, x, epsilons):
        return self.model_autoencoder.predict([x, epsilons], verbose=0, batch_size=self.batch_size)

    def repeat_x(self, x):
        x = Reshape((self.minibatch_size, 1, self.input_dim))(x)
        x = RepeatVectorAxis(self.t, axis=2)(x)
        return x


    def create_model_encoder(self, hidden_dim=128, reg=lambda: l1(1e-6)):
        input_x = Input(shape=(self.minibatch_size, self.input_dim), name="input_x_encoder")
        input_epsilon = Input(shape=(self.minibatch_size, self.t, self.latent_dim), name="input_epsilon_encoder")
        enc_lstm = LSTM(hidden_dim, return_sequences=True, name="encoder_lstm", W_regularizer=reg(),
                        U_regularizer=reg())
        enc_z_mu = Dense(self.latent_dim, activation='linear', W_regularizer=reg(), name="enc_z_mu")
        enc_z_log_sigma_sq = Dense(self.latent_dim, activation='linear', W_regularizer=reg(), name="enc_z_log_sigma_sq")

        h = self.repeat_x(input_x)
        h = TimeDistributed(enc_lstm)(h)
        _enc_z_mu = TimeDistributed(TimeDistributed(enc_z_mu))(h)
        _enc_z_log_sima_sq = TimeDistributed(TimeDistributed(enc_z_log_sigma_sq))(h)
        _enc_z = merge([_enc_z_mu, _enc_z_log_sima_sq, input_epsilon],
                       mode=lambda (a2, b2, c2): a2 + (K.exp(b2 / 2) * c2),
                       output_shape=lambda d2: d2[0])
        self.model_encoder = Model([input_x, input_epsilon], _enc_z, name="encoder")

        input_x_single = Input(shape=(self.input_dim,))
        input_epsilon_single = Input(shape=(self.t, self.latent_dim))
        h = input_x_single
        h = Reshape((1, self.input_dim))(h)
        h = RepeatVectorAxis(self.t, axis=1)(h)
        h = enc_lstm(h)
        _enc_z_mu = TimeDistributed(enc_z_mu)(h)
        _enc_z_log_sima_sq = TimeDistributed(enc_z_log_sigma_sq)(h)
        _enc_z = merge([_enc_z_mu, _enc_z_log_sima_sq, input_epsilon_single],
                       mode=lambda (a1, b1, c1): a1 + (K.exp(b1 / 2) * c1),
                       output_shape=lambda d1: d1[0])
        self.model_encoder_single = Model([input_x_single, input_epsilon_single], _enc_z, name="encoder_single")

    def create_model_decoder(self, hidden_dim=128, reg=lambda: l1(1e-6)):
        input_z = Input(shape=(self.minibatch_size, self.t, self.latent_dim), name="input_z_decoder")
        dec_lstm = LSTM(hidden_dim, return_sequences=True, name="decoder_lstm", W_regularizer=reg(),
                        U_regularizer=reg())
        dec_h = Dense(self.input_dim, activation='linear', W_regularizer=reg())
        dec_c = Lambda(cumsum, output_shape=lambda a: a)
        h = input_z
        h = TimeDistributed(dec_lstm)(h)
        h = TimeDistributed(TimeDistributed(dec_h))(h)
        h = dec_c(h)
        _dec_x = Activation('sigmoid')(h)
        self.model_decoder = Model([input_z], _dec_x, name="decoder")

        input_z_single = Input(shape=(self.t, self.latent_dim))
        h = input_z_single
        h = dec_lstm(h)
        h = TimeDistributed(dec_h)(h)
        dec_c = Lambda(cumsum_single, output_shape=lambda a: a)
        h = dec_c(h)
        _dec_x = Activation('sigmoid')(h)
        self.model_decoder_single = Model([input_z_single], _dec_x, name="decoder")

    def create_model_discriminator(self, hidden_dim=128, reg=lambda: l1(1e-6), dropout=0.5):
        input_x = Input(shape=(self.minibatch_size, self.t, self.input_dim), name="input_x_discriminator")
        input_z = Input(shape=(self.minibatch_size, self.t, self.latent_dim), name="input_z_discriminator")

        disc_lstm_z = LSTM(hidden_dim, return_sequences=True, name="discriminator_lstm_z", W_regularizer=reg(),
                           U_regularizer=reg(), dropout_U=dropout, dropout_W=dropout)
        disc_hx = []
        disc_hx.append(TimeDistributed(
            TimeDistributed(Dense(hidden_dim, activation='tanh', name="disc_hx1", W_regularizer=reg()))))
        disc_hx.append(Dropout(dropout))
        disc_hx.append(TimeDistributed(
            TimeDistributed(Dense(hidden_dim, activation='tanh', name="disc_hx2", W_regularizer=reg()))))
        disc_hx.append(Dropout(dropout))

        disc_lstm_y_forwards = LSTM(hidden_dim, return_sequences=True, name="disc_lstm_y_forwards",
                                    W_regularizer=reg(), U_regularizer=reg(), dropout_U=dropout, dropout_W=dropout)
        dense_y = Dense(1, activation='sigmoid', name='disc_y', W_regularizer=reg())

        hz = input_z
        hz = TimeDistributed(disc_lstm_z)(hz)
        hx = input_x
        hx = apply_layers(hx, disc_hx)
        h = merge([hz, hx], mode='concat', concat_axis=-1)
        h = Permute((2, 1, 3))(h)
        h = TimeDistributed(disc_lstm_y_forwards)(h)
        y = TimeDistributed(TimeDistributed(dense_y))(h)
        self.model_discriminator = Model([input_x, input_z], y, name="discriminator")

    def create_models(self):
        self.create_model_encoder()
        self.create_model_decoder()
        self.create_model_discriminator()

        self.model_encoder.summary()
        self.model_decoder.summary()
        self.model_discriminator.summary()

        input_z = Input(shape=(self.minibatch_size, self.t, self.latent_dim), name="input_z")
        input_x = Input(shape=(self.minibatch_size, self.input_dim,), name="input_x")
        input_epsilon = Input(shape=(self.minibatch_size, self.t, self.latent_dim), name="input_epsilon")

        x_rep = self.repeat_x(input_x)
        z_real = self.model_encoder([input_x, input_epsilon])
        x_fake = self.model_decoder([input_z])
        y_real = Activation('linear', name='y_real')(self.model_discriminator([x_rep, z_real]))
        y_fake = Activation('linear', name='y_fake')(self.model_discriminator([x_fake, input_z]))

        loss = "binary_crossentropy"
        losses = {"y_real": loss, "y_fake": loss}

        self.set_trainable(is_generator=True)
        self.model_gan_generator = Model([input_x, input_z, input_epsilon], [y_fake, y_real])
        self.model_gan_generator.compile(optimizer=self.opt_generator, loss=losses)

        self.set_trainable(is_generator=False)
        self.model_gan_discriminator = Model([input_x, input_z, input_epsilon], [y_fake, y_real])
        self.model_gan_discriminator.compile(optimizer=self.opt_discriminator, loss=losses)

        input_x_single = Input(shape=(self.input_dim,), name="input_x")
        input_epsilon_single = Input(shape=(self.t, self.latent_dim), name="input_epsilon")
        self.model_autoencoder = Model([input_x_single, input_epsilon_single],
                                       self.model_decoder_single(
                                           self.model_encoder_single([input_x_single, input_epsilon_single])))

        self.models = {"encoder": self.model_encoder,
                       "decoder": self.model_decoder,
                       "discriminator": self.model_discriminator
                       }

    def summary(self):
        self.set_trainable(is_generator=True)
        print("model_encoder")
        self.model_encoder.summary()
        print("model_decoder")
        self.model_decoder.summary()
        self.set_trainable(is_generator=False)
        print("model_discriminator")
        self.model_discriminator.summary()
        self.set_trainable(is_generator=False)
        print("model_gan_discriminator")
        self.model_gan_discriminator.summary()
        self.set_trainable(is_generator=True)
        print("model_gan_generator")
        self.model_gan_generator.summary()

    def train(self, x_train, x_test, path, nb_epoch, nb_batch=1000):
        self.log = []
        z_samples = self.prior_sample(100)
        x_samples = n_choice(x_test, 10)
        self.test_epoch(path="%s-initial" % path, x_train=x_train, x_test=x_test, x_samples=x_samples,
                        z_samples=z_samples)
        x_train = np.copy(x_train)
        for epoch in tqdm(range(nb_epoch), desc="Epoch"):
            np.random.shuffle(x_train)
            self.train_epoch(x_train, nb_batch)
            self.test_epoch(path="%s-epoch-%03i" % (path, epoch), x_train=x_train, x_test=x_test, x_samples=x_samples,
                            z_samples=z_samples)
            if epoch % 10 == 0:
                self.save(file="%s-checkpoint-%03i" % (path, epoch))

    def train_epoch(self, x_train, nb_batch):
        for batch in tqdm(range(nb_batch), desc="Batch"):
            for i in range(4):
                x = n_choice(x_train, self.batch_size)
                x = x.reshape((-1, self.minibatch_size, 28 * 28))
                n = x.shape[0]
                zs = self.prior_samples(n)
                epsilons = self.epsilon_samples(n)
                self.train_batch_d(x, zs, epsilons)
            x = n_choice(x_train, self.batch_size)
            x = x.reshape((-1, self.minibatch_size, 28 * 28))
            n = x.shape[0]
            zs = self.prior_samples(n)
            epsilons = self.epsilon_samples(n)
            self.train_batch_g(x, zs, epsilons)


    def train_batch(self, x):
        x = x.reshape((-1, self.minibatch_size, 28 * 28))
        n = x.shape[0]
        zs = self.prior_samples(n)
        epsilons = self.epsilon_samples(n)
        self.train_batch_d(x, zs, epsilons)
        self.train_batch_g(x, zs, epsilons)


    def test_batch(self, x, z, epsilon):
        self.set_trainable(is_generator=False)
        y_fake, y_real = self.model_gan_discriminator.predict_on_batch([x, z, epsilon])
        margin = 0.1
        y_fake_incorrect = np.count_nonzero(y_fake > (0.5 - margin))
        y_real_incorrect = np.count_nonzero(y_real < (0.5 + margin))
        return (y_fake_incorrect == 0) and (y_real_incorrect == 0)

    def train_batch_d(self, x, zs, epsilons):
        n = x.shape[0]
        self.set_trainable(is_generator=False)
        return self.model_gan_discriminator.train_on_batch([x, zs, epsilons], self.target_d(n))

    def train_batch_g(self, x, zs, epsilons):
        n = x.shape[0]
        self.set_trainable(is_generator=True)
        return self.model_gan_generator.train_on_batch([x, zs, epsilons], self.target_g(n))

    def target_g(self, n):
        shape = (n, self.t, self.minibatch_size, 1)
        y_fake = np.ones(shape)
        y_real = np.zeros(shape)
        return [y_fake, y_real]

    def target_d(self, n, smoothing=False):
        shape = (n, self.t, self.minibatch_size, 1)
        if smoothing:
            y_fake = np.ones(shape) * 0.0
            y_real = np.ones(shape) * 0.9
            return [y_fake, y_real]
        else:
            y_fake = np.zeros(shape)
            y_real = np.ones(shape)
            return [y_fake, y_real]

    def prior_samples(self, n):
        return np.random.normal(loc=0.0, scale=1.0, size=(n, self.minibatch_size, self.t, self.latent_dim))

    def prior_sample(self, n):
        return np.random.normal(loc=0.0, scale=1.0, size=(n, self.t, self.latent_dim))

    def epsilon_samples(self, n):
        return np.random.normal(loc=0.0, scale=1.0, size=(n, self.minibatch_size, self.t, self.latent_dim))

    def epsilon_sample(self, n):
        return np.random.normal(loc=0.0, scale=1.0, size=(n, self.t, self.latent_dim))

        ###############################
        # Testing
        ###############################

    def test_epoch(self, x_train, x_test, path, x_samples, z_samples):
        self.save_figures(path, x_samples, z_samples)
        # logging.info("Testing")
        train = self.test_metrics(x_train)
        test = self.test_metrics(x_test)
        self.print_metrics("Train", train)
        self.print_metrics("Test", test)
        metrics = {"train": train, "test": test}
        self.log.append(metrics)
        return metrics

    def print_metrics(self, name, m):
        fmt = "%s loss: %f, generator: %f (%f/%f), discriminator: %f (%f/%f)"
        logging.info(fmt % (name, m["total"]["loss"],
                            m["generator"]["loss"], m["generator"]["y_real_loss"], m["generator"]["y_fake_loss"],
                            m["discriminator"]["loss"], m["discriminator"]["y_real_loss"],
                            m["discriminator"]["y_fake_loss"]))

    def save_figures(self, path, x_samples, z_samples):
        self.save_figures_generated(path, z_samples)
        self.save_figures_autoencoded(path, x_samples)
        self.save_figures_drawing(path, x_samples)

    def save_figures_generated(self, path, z_samples):
        samples = self.decode(z_samples)[:, -1, :]
        fig = show_samples(samples=samples, dim=(10, 10))
        fig.savefig("%s-generated.png" % path)
        plt.close(fig)

    def save_figures_autoencoded(self, path, x_samples):
        m = 10
        epsilons = self.epsilon_sample(x_samples.shape[0] * m)
        autoencoded = self.autoencode(np.repeat(x_samples, m, axis=0), epsilons)[:, -1, :]
        fig = show_autoencoding(x_samples.reshape((-1, 1, 28, 28)), autoencoded.reshape((-1, 1, 28, 28)))
        fig.savefig("%s-autoencoded.png" % path)
        plt.close(fig)

    def save_figures_drawing(self, path, x_samples):
        n = x_samples.shape[0]
        epsilons = self.epsilon_sample(n)
        autoencoded = self.autoencode(x_samples, epsilons)
        fig = show_autoencoding(x_samples.reshape((-1, 1, 28, 28)), autoencoded.reshape((-1, 1, 28, 28)),
                                figsize=(self.t + 1, n))
        fig.savefig("%s-drawing.png" % path)
        plt.close(fig)

    def test_metrics(self, x, samples=100):
        x = n_choice(x, samples * self.minibatch_size)
        x = x.reshape((samples, self.minibatch_size, 28 * 28))
        # x = self.repeat_x_samples(x)
        n = x.shape[0]
        zs = self.prior_samples(n)
        epsilons = self.epsilon_samples(n)

        # Test generator
        self.set_trainable(is_generator=True)
        m1 = self.model_gan_generator.evaluate([x, zs, epsilons], self.target_g(n), batch_size=self.batch_size,
                                               verbose=0)
        m1 = parse_metrics(m1, self.model_gan_generator)

        # Test discriminator
        self.set_trainable(is_generator=False)
        m2 = self.model_gan_discriminator.evaluate([x, zs, epsilons], self.target_d(n), batch_size=self.batch_size,
                                                   verbose=0)
        m2 = parse_metrics(m2, self.model_gan_discriminator)

        return {"generator": m1, "discriminator": m2, "total": {"loss": m1["loss"] + m2["loss"]}}

    def save(self, file):
        for name, model in self.models.iteritems():
            model.save_weights("%s-%s.h5" % (file, name))
        if self.log:
            with open("%s-log.pickle" % file, "wb") as f:
                pickle.dump(self.log, f)

    def load(self, file):
        for name, model in self.models.iteritems():
            model.load_weights("%s-%s.h5" % (file, name))
        if os.path.exists(file + "-log.pickle"):
            with open("%s-log.pickle" % file, "rb") as f:
                self.log = pickle.load(f)


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/bigran-lstm"
    if not os.path.exists(path):
        os.makedirs(path)
    path = "%s/bigran" % path

    print("Creating model")
    model = BiGRAN.mnist_model()
    model.summary()

    print("Loading data")
    x_train, y_train, x_test, y_test = mnist_data()

    print("Training")
    model.train(x_train=x_train, x_test=x_test, nb_epoch=300, path=path)

    print("Saving")
    model.save(path)
