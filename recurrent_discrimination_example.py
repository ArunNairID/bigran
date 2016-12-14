
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

from keras.layers import Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm

def create_discriminator(input_dim, depth):
    x = Input(shape=(depth, input_dim))
    lstm = LSTM(256, return_sequences=True)(x)
    y = TimeDistributed(Dense(1, activation='sigmoid'))(lstm)
    m = Model(x, y, name='discriminator')
    return m

def create_discriminator_pair(input_dim, depth, discriminator):
    xfake = Input(shape=(depth, input_dim))
    xreal = Input(shape=(depth, input_dim))
    yfake = discriminator(xfake)
    yreal = discriminator(xreal)
    m = Model([xfake, xreal],[yfake, yreal], name='discriminator_pair')
    return m

def one_hot(labels, m):
    labels = np.array(labels).ravel()
    n = labels.shape[0]
    y = np.zeros((n,m))
    y[np.arange(n), labels]=1
    return y

def x_fake(batch_size):
    return one_hot(np.random.randint(0, 5, size=(batch_size,)), 10)

def x_real(batch_size):
    return one_hot(np.random.randint(0,10, size=(batch_size,)), 10)

def test_sequence(discriminator, seq):
    x = one_hot(seq, 10).reshape((1,len(seq),10))
    y = discriminator.predict(x)
    print "Samples: %s"%str(seq)
    print "P(real): %s"%str(y.ravel())

def test_rnn(depth, sequences):

    input_dim = 10
    nb_epoch = 10000
    batch_size = 128
    m = 10


    discriminator = create_discriminator(input_dim, depth)
    discriminator_pair = create_discriminator_pair(input_dim, depth, discriminator)
    discriminator_pair.compile(Adam(1e-4, decay=1e-3),'binary_crossentropy')
    discriminator.summary()
    discriminator_pair.summary()
    for i in tqdm(range(nb_epoch)):
        _x_fake = x_fake(batch_size).reshape((-1,depth, m))
        _x_real = x_real(batch_size).reshape((-1,depth, m))
        discriminator_pair.train_on_batch([_x_fake, _x_real],
                                          [np.zeros((batch_size/depth, depth, 1)), np.ones((batch_size/depth, depth, 1))])
    for sequence in sequences:
        test_sequence(discriminator, sequence)

test_rnn(4, [[0,0,0,0], [1,2,3,4], [1,2,3,5], [5,1,2,3]])
test_rnn(2, [[0,0], [0,5], [5,0]])
test_rnn(1, [[0],[1],[2],[5],[6]])

"""
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 4, 10)         0
____________________________________________________________________________________________________
lstm_1 (LSTM)                    (None, 4, 256)        273408      input_1[0][0]
____________________________________________________________________________________________________
timedistributed_1 (TimeDistribute(None, 4, 1)          257         lstm_1[0][0]
====================================================================================================
Total params: 273665
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, 4, 10)         0
____________________________________________________________________________________________________
input_3 (InputLayer)             (None, 4, 10)         0
____________________________________________________________________________________________________
discriminator (Model)            (None, 4, 1)          273665      input_2[0][0]
                                                                   input_3[0][0]
====================================================================================================
Total params: 273665
____________________________________________________________________________________________________

Samples: [0, 0, 0, 0]
P(real): [ 0.33631474  0.19826342  0.1088916   0.04864468]
Samples: [1, 2, 3, 4]
P(real): [ 0.33726114  0.19887531  0.12053937  0.08286883]
Samples: [1, 2, 3, 5]
P(real): [ 0.33726114  0.19887531  0.12053937  0.99992919]
Samples: [5, 1, 2, 3]
P(real): [ 0.99839407  1.          1.          1.        ]
Samples: [0, 0]
P(real): [ 0.33647439  0.21365139]
Samples: [0, 5]
P(real): [ 0.33647439  0.99998987]
Samples: [5, 0]
P(real): [ 0.99946231  1.        ]
Samples: [0]
P(real): [ 0.33677009]
Samples: [1]
P(real): [ 0.325082]
Samples: [2]
P(real): [ 0.33696172]
Samples: [5]
P(real): [ 0.99984622]
Samples: [6]
P(real): [ 0.99983513]
"""
