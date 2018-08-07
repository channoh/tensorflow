#!/usr/bin/env python3

#
# https://www.tensorflow.org/versions/r1.0/get_started/mnist/pros
#

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes
import util
import tensorflow as tf
import time
import collections
import numpy as np

def network(mnist):
    # mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)
    
    X = tf.placeholder(tf.float32, [None, 784])         # input image == 28x28 pixels
    X_im = tf.reshape(X, [-1, 28, 28, 1])               # reshape input image for convolution
    Y = tf.placeholder(tf.float32, [None, 10])          # one hot vector of output (0~9)
    keep_prob = tf.placeholder(tf.float32)              # dropout

    # first convolutional layer: compute 32 features for each 5x5 patch
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b1 = tf.constant(0.1, shape=[32])
    L1 = tf.nn.conv2d(X_im, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1 + b1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # second convolutional layer
    W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b2 = tf.constant(0.1, shape=[64])
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2 + b2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully connected layer
    W3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b3 = tf.constant(0.1, shape=[1024])
    L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
    L3 = tf.nn.relu(tf.matmul(L3, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob)

    # fully connected layer
    W4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b4 = tf.constant(0.1, shape=[10])
    pred_Y = tf.matmul(L3, W4) + b4

    # define an optimizer to reduce the loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # 
    is_correct = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # execute the optimizer
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # train
        print(mnist.train.num_examples)
        print(mnist.validation.num_examples)
        print(mnist.test.num_examples)
        start = time.time()
        batch_size = 50
        max_iter = int(mnist.train.num_examples / batch_size)
        for epoch in range(200):
            for i in range(max_iter):
                batch_xs = mnist.train.images[i * batch_size : (i+1) * batch_size]
                batch_ys = mnist.train.labels[i * batch_size : (i+1) * batch_size]
                if i % 100 == 0:
                    # evaluate
                    accuracy_val = accuracy.eval(feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
                    print("step %d: accuracy= %.4f, %.2fs" % (i + epoch * max_iter, accuracy_val, time.time() - start))
                    start = time.time()
                optimizer.run(feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})

        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                Y: mnist.test.labels,
                                                keep_prob: 1.0})
        print("accuracy:", accuracy_val)


def main():
    DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    DATA_DIR_PATH = './data/MNIST'
    VALIDATION_SIZE = 5000

    local_file = util.maybe_download(TRAIN_IMAGES, DATA_DIR_PATH, DEFAULT_SOURCE_URL + TRAIN_IMAGES)
    train_images = util.extract_images(local_file)

    local_file = util.maybe_download(TRAIN_LABELS, DATA_DIR_PATH, DEFAULT_SOURCE_URL + TRAIN_LABELS)
    train_labels = util.extract_labels(local_file)

    local_file = util.maybe_download(TEST_IMAGES,  DATA_DIR_PATH, DEFAULT_SOURCE_URL + TEST_IMAGES)
    test_images = util.extract_images(local_file)

    local_file = util.maybe_download(TEST_LABELS,  DATA_DIR_PATH, DEFAULT_SOURCE_URL + TEST_LABELS)
    test_labels = util.extract_labels(local_file)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    Dataset = collections.namedtuple('Dataset', ['images', 'labels', 'num_examples'])
    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    # train images
    num_train_images = train_images.shape[0]
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]).astype(np.float32)
    train_images = np.multiply(train_images, 1.0 / 255.0) # Convert from [0, 255] -> [0.0, 1.0].
    train = Dataset(train_images, train_labels, num_train_images)

    # validation images
    num_validation_images = validation_images.shape[0]
    validation_images = validation_images.reshape(validation_images.shape[0], validation_images.shape[1] * validation_images.shape[2]).astype(np.float32)
    validation_images = np.multiply(validation_images, 1.0 / 255.0) # Convert from [0, 255] -> [0.0, 1.0].
    validation = Dataset(validation_images, validation_labels, num_validation_images)

    # test images
    num_test_images = test_images.shape[0]
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2]).astype(np.float32)
    test_images = np.multiply(test_images, 1.0 / 255.0) # Convert from [0, 255] -> [0.0, 1.0].
    test = Dataset(test_images, test_labels, num_test_images)

    mnist_data = Datasets(train=train, validation=validation, test=test)

    network(mnist_data)

if __name__ == "__main__":
    main()
