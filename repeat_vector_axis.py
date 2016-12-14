
from keras.layers import Layer
import keras.backend as K


class RepeatVectorAxis(Layer):
    '''Repeats the input n times.
    # Example
    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension
        model.add(RepeatVector(3))
        # now: model.output_shape == (None, 3, 32)
    ```
    # Arguments
        n: integer, repetition factor.
    # Input shape
        2D tensor of shape `(nb_samples, features)`.
    # Output shape
        3D tensor of shape `(nb_samples, n, features)`.
    '''

    def __init__(self, n, axis=1, **kwargs):
        self.n = n
        self.axis=axis
        self.input_spec = []
        super(RepeatVectorAxis, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        print "Input shape: %s"%str(input_shape)
        shape = list(input_shape)
        shape[self.axis] *= self.n
        print "get_output_shape_for shape: %s"%str(shape)
        return tuple(shape)

    def call(self, x, mask=None):
        return K.repeat_elements(x, self.n, self.axis)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatVectorAxis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    from keras.models import Model
    from keras.layers import Input, Reshape

    x = Input(shape=(10,))
    h = Reshape((1, 10))(x)
    h = RepeatVectorAxis(n=4, axis=1)(h)
    m = Model(x, h)
    m.summary()

    x = Input(shape=(10, 3))
    h = Reshape((10, 1, 3))(x)
    h = RepeatVectorAxis(n=4, axis=2)(h)
    m = Model(x, h)
    m.summary()

    x = Input(shape=(10, 3))
    h = Reshape((10, 3, 1))(x)
    h = RepeatVectorAxis(n=4, axis=3)(h)
    m = Model(x, h)
    m.summary()

    import numpy as np

    x = Input(shape=(2, 3))
    h = Reshape((2, 3, 1))(x)
    h = RepeatVectorAxis(n=4, axis=3)(h)
    m = Model(x, h)
    m.summary()
    _x = np.random.random((1,2,3))
    print _x
    print m.predict(_x)

