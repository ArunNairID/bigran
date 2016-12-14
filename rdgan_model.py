import matplotlib as mpl

mpl.use('Agg')
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

from bigran_model import n_choice, apply_layers, model_set_trainable, parse_metrics


class RDGRAN(object):
    @classmethod
    def mnist_model(cls,
                    latent_dim=100, input_dim=28 * 28, batch_size=64,
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
                    minibatch_size=minibatch_size, decay_d=decay_d, decay_g=decay_g
                    )
        return model

    def set_trainable(self, is_generator):
        model_set_trainable(self.model_decoder, is_generator)
        model_set_trainable(self.model_discriminator, not is_generator)

    def __init__(self, input_dim, latent_dim, batch_size, lr_d, lr_g, minibatch_size, decay_d, decay_g):
        self.minibatch_size = minibatch_size
        self.opt_discriminator = Adam(lr=lr_d, decay=decay_d)
        self.opt_generator = Adam(lr=lr_g, decay=decay_g)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.create_models()
        self.log = None

    def decode(self, zs):
        return self.model_decoder_single.predict(zs, verbose=0, batch_size=self.batch_size)

    def create_model_decoder(self, hidden_dim=256, reg=lambda: l1(1e-6)):
        input_z = Input(shape=(self.minibatch_size, self.latent_dim), name="input_z_decoder")

        h1 = Dense(hidden_dim, activation='tanh', name="decoder_h1", W_regularizer=reg())
        h2 = Dense(hidden_dim, activation='tanh', name="decoder_h2", W_regularizer=reg())
        x = Dense(self.input_dim, activation='sigmoid', name="decoder_x", W_regularizer=reg())

        dec_h = [TimeDistributed(h1), TimeDistributed(h2), TimeDistributed(x)]
        _dec_x = apply_layers(input_z, dec_h)
        self.model_decoder = Model([input_z], _dec_x, name="decoder")

        input_z_single = Input(shape=(self.latent_dim,))
        dec_h = [h1,h2,x]
        _dec_x = apply_layers(input_z_single, dec_h)
        self.model_decoder_single = Model([input_z_single], _dec_x, name="decoder")

    def create_model_discriminator(self, hidden_dim=256, reg=lambda: l1(1e-5)):
        input_x = Input(shape=(self.minibatch_size, self.input_dim), name="input_x_discriminator")

        disc_lstm = LSTM(hidden_dim, return_sequences=True, name="discriminator_lstm", W_regularizer=reg(),
                           U_regularizer=reg())
        dense_y = Dense(1, activation='sigmoid', name='disc_y', W_regularizer=reg())

        h = Bidirectional(disc_lstm, merge_mode='concat')(input_x)
        y = TimeDistributed(dense_y)(h)
        self.model_discriminator = Model([input_x], y, name="discriminator")

    def create_models(self):
        self.create_model_decoder()
        self.create_model_discriminator()

        self.model_decoder.summary()
        self.model_discriminator.summary()

        input_z = Input(shape=(self.minibatch_size, self.latent_dim), name="input_z")
        input_x = Input(shape=(self.minibatch_size, self.input_dim,), name="input_x")

        x_fake = self.model_decoder([input_z])
        # x_rep = self.repeat_x(input_x)
        y_real = Activation('linear', name='y_real')(self.model_discriminator([input_x]))
        y_fake = Activation('linear', name='y_fake')(self.model_discriminator([x_fake]))

        loss = "binary_crossentropy"
        losses = {"y_real": loss, "y_fake": loss}

        self.set_trainable(is_generator=True)
        self.model_gan_generator = Model([input_z], [y_fake])
        self.model_gan_generator.compile(optimizer=self.opt_generator, loss=loss)

        self.set_trainable(is_generator=False)
        self.model_gan_discriminator = Model([input_x, input_z], [y_fake, y_real])
        self.model_gan_discriminator.compile(optimizer=self.opt_discriminator, loss=losses)

        self.models = {
                       "decoder": self.model_decoder,
                       "discriminator": self.model_discriminator
                       }

    def summary(self):
        self.set_trainable(is_generator=True)
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
        self.test_epoch(path="%s-initial" % path, x_train=x_train, x_test=x_test,
                        z_samples=z_samples)
        x_train = np.copy(x_train)
        for epoch in tqdm(range(nb_epoch), desc="Epoch"):
            np.random.shuffle(x_train)
            self.train_epoch(x_train, nb_batch)
            self.test_epoch(path="%s-epoch-%03i" % (path, epoch), x_train=x_train, x_test=x_test,
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
                self.train_batch_d(x, zs)
            x = n_choice(x_train, self.batch_size)
            x = x.reshape((-1, self.minibatch_size, 28 * 28))
            n = x.shape[0]
            zs = self.prior_samples(n)
            self.train_batch_g(x, zs)
        #self.train_batch(x)

    def train_batch(self, x):
        x = x.reshape((-1, self.minibatch_size, 28 * 28))
        n = x.shape[0]
        zs = self.prior_samples(n)
        self.train_batch_d(x, zs)
        self.train_batch_g(x, zs)


    def test_batch(self, x, z):
        self.set_trainable(is_generator=False)
        y_fake, y_real = self.model_gan_discriminator.predict_on_batch([x, z])
        margin = 0.1
        y_fake_incorrect = np.count_nonzero(y_fake > (0.5 - margin))
        y_real_incorrect = np.count_nonzero(y_real < (0.5 + margin))
        return (y_fake_incorrect == 0) and (y_real_incorrect == 0)

    def train_batch_d(self, x, zs):
        n = x.shape[0]
        self.set_trainable(is_generator=False)
        return self.model_gan_discriminator.train_on_batch([x, zs], self.target_d(n))

    def train_batch_g(self, x, zs):
        n = x.shape[0]
        self.set_trainable(is_generator=True)
        return self.model_gan_generator.train_on_batch([zs], self.target_g(n))

    def target_g(self, n):
        shape = (n, self.minibatch_size, 1)
        y_fake = np.ones(shape)
        return [y_fake]

    def target_d(self, n, smoothing=True):
        shape = (n, self.minibatch_size, 1)
        if smoothing:
            y_fake = np.ones(shape) * 0.0
            y_real = np.ones(shape) * 0.9
            return [y_fake, y_real]
        else:
            y_fake = np.zeros(shape)
            y_real = np.ones(shape)
            return [y_fake, y_real]

    def prior_samples(self, n):
        return np.random.normal(loc=0.0, scale=1.0, size=(n, self.minibatch_size, self.latent_dim))

    def prior_sample(self, n):
        return np.random.normal(loc=0.0, scale=1.0, size=(n, self.latent_dim))


        ###############################
        # Testing
        ###############################

    def test_epoch(self, x_train, x_test, path, z_samples):
        self.save_figures(path,z_samples)
        train = self.test_metrics(x_train)
        test = self.test_metrics(x_test)
        self.print_metrics("Train", train)
        self.print_metrics("Test", test)
        metrics = {"train": train, "test": test}
        self.log.append(metrics)
        return metrics

    def print_metrics(self, name, m):
        fmt = "%s loss: %f, generator: %f, discriminator: %f (%f/%f)"
        logging.info(fmt % (name, m["total"]["loss"],
                            m["generator"]["loss"],
                            m["discriminator"]["loss"], m["discriminator"]["y_real_loss"],
                            m["discriminator"]["y_fake_loss"]))

    def save_figures(self, path, z_samples):
        self.save_figures_generated(path, z_samples)

    def save_figures_generated(self, path, z_samples):
        samples = self.decode(z_samples) #[:, -1, :]
        fig = show_samples(samples=samples, dim=(10, 10))
        fig.savefig("%s-generated.png" % path)
        plt.close(fig)


    def test_metrics(self, x, samples=100):
        x = n_choice(x, samples * self.minibatch_size)
        x = x.reshape((samples, self.minibatch_size, 28 * 28))
        # x = self.repeat_x_samples(x)
        n = x.shape[0]
        zs = self.prior_samples(n)

        # Test generator
        self.set_trainable(is_generator=True)
        m1 = self.model_gan_generator.evaluate([zs], self.target_g(n), batch_size=self.batch_size,
                                               verbose=0)
        m1 = parse_metrics(m1, self.model_gan_generator)

        # Test discriminator
        self.set_trainable(is_generator=False)
        m2 = self.model_gan_discriminator.evaluate([x, zs], self.target_d(n), batch_size=self.batch_size,
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
    path = "output/rdgan/test"
    if not os.path.exists(path):
        os.makedirs(path)
    path = "%s/rdgran" % path

    print("Creating model")
    model = RDGRAN.mnist_model()
    model.summary()

    print("Loading data")
    x_train, y_train, x_test, y_test = mnist_data()

    print("Training")
    model.train(x_train=x_train, x_test=x_test, nb_epoch=300, path=path)

    print("Saving")
    model.save(path)
