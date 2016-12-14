import logging.config
import logging
import os
from bigran_model import BiGRAN, mnist_data

if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/bigran-lstm-polished"
    if not os.path.exists(path):
        os.makedirs(path)
    path = "%s/bigran" % path

    print("Creating model")
    model = BiGRAN.mnist_model(lr_d=3e-4, lr_g=1e-4, decay_d=1e-4, decay_g=4e-4)
    model.summary()
    modelin = "bigran-lstm-bak/bigran-checkpoint-120"
    model.load(modelin)

    print("Loading data")
    x_train, y_train, x_test, y_test = mnist_data()

    print("Training")
    model.train(x_train=x_train, x_test=x_test, nb_epoch=300, path=path)

    print("Saving")
    model.save(path)