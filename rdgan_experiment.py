from rdgan_model import RDGRAN
import logging.config
from rdgan_model import mnist_data
import os


def experiment(path, minibatch_size):

    model = RDGRAN.mnist_model(minibatch_size=minibatch_size)
    model.summary()

    print("Loading data")
    x_train, y_train, x_test, y_test = mnist_data()

    print("Training")
    model.train(x_train=x_train, x_test=x_test, nb_epoch=51, path=path)

    print("Saving")
    model.save(path)


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/rdgan/experiment"
    if not os.path.exists(path):
        os.makedirs(path)

    sizes = [1, 2, 4, 8]
    for size in sizes:
        testpath = os.path.join(path, "minibatch-%i" % size)
        if not os.path.exists(testpath):
            os.makedirs(testpath)
            testpath = os.path.join(testpath, "rdgan")
            experiment(testpath, size)
