# bigran
Bi-Directional Generative Recurrent Adversarial Network

Software requires Python2.7, keras, and Theano or TensorFlow
BiGRAN works on Theano (CPU only) and TensorFlow, but crashes Theano (GPU) optimizer.

recurrent_discrimination_example.py
Train discriminator on toy problem 

rdgan_experiment.py
Train recurrent discriminator GAN with several depths

rdgan_model.py
Model for recurrent discriminator experiments

bigran_model.py
Train BiGRAN model on MNIST dataset

bigran_model_polish.py
Train BiGRAN model from checkpoint with different learning rates

repeat_vector_axis.py
Keras layer to repeat vectors along an axis

show_autoencoding.py
Helper to save autoencoding images

show_samples.py
Helper to save grids of generated images

mnist_data.py
Helper to process MNIST digits