import os
import model
import tensorflow as tf
import input_data

data = input_data.read_data_sets('MNIST_data', one_hot=True)
