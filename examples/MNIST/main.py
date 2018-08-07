#!/usr/bin/env python3


#
# https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners
#

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import os

class MNIST:
    sess = None
    X = None
    Y = None
    keep_prob = None
    model = None
    loss = None
    train_op = None
    mnist = None

    def __init__(self):
        self.mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)
        self.X = tf.placeholder(tf.float32, [None, 784]) # input image == 28x28 pixels == 784
        self.Y = tf.placeholder(tf.float32, [None, 10])  # one hot vector of output (0~9)
        self.keep_prob = tf.placeholder(tf.float32)      # dropout

    def init_session(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def build_model(self):
        X_im = tf.reshape(self.X, [-1, 28, 28, 1])               # reshape input image for convolution
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
        L3 = tf.nn.dropout(L3, self.keep_prob)

        # fully connected layer
        W4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b4 = tf.constant(0.1, shape=[10])
        self.model = tf.add(tf.matmul(L3, W4), b4, name='model')

    def build_training_op(self):
        # define an optimizer to reduce the loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.model))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def save_model(self, model_path="./model/"):
        if not os.path.exists(model_path):
            print("create dir {}".format(model_path))
            os.makedirs(model_path)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, model_path + "model.ckpt")
        print('Saving the model to {}...'.format(model_path))

    def load_model(self, model_path="./model/"):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path + "model.ckpt")
        print('Loading the model from {}...'.format(model_path))

    def train(self):
        for i in range(20000):
            batch_xs, batch_ys = self.mnist.train.next_batch(50)
            feed_dict = {self.X: batch_xs, self.Y: batch_ys, self.keep_prob: 0.5}
            _, loss_val = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            if i % 100 == 0:
                print("step %3d: loss = %.4f" % (i, loss_val))

    def eval(self):
        is_correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        feed_dict = {self.X: self.mnist.test.images, self.Y: self.mnist.test.labels, self.keep_prob: 1.0}
        print("accuracy:", self.sess.run(accuracy, feed_dict=feed_dict))

    def infer(self, image):
        feed_dict = {self.X: image.reshape(-1, 784), self.keep_prob: 1.0}
        number = self.sess.run(tf.argmax(self.model, 1), feed_dict=feed_dict)[0]
        accuracy = self.sess.run(tf.nn.softmax(self.model), feed_dict=feed_dict)[0]
        return (accuracy[number], number)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--train', dest='train', action='store_true')
    args.add_argument('--infer', dest='train', action='store_false')
    args.set_defaults(train = True)
    config = args.parse_args()

    if config.train:    ## training
        mnist = MNIST()
        mnist.build_model()
        mnist.build_training_op()
        mnist.init_session()
        mnist.train()
        mnist.save_model()
        mnist.eval()

    else:               ## inference
        mnist = MNIST()
        mnist.build_model()
        mnist.init_session()
        mnist.load_model()
        test_image = mnist.mnist.test.images[0]
        print(mnist.infer(test_image))
