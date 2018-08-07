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
    model = None
    loss = None
    train_op = None
    mnist = None

    def __init__(self):
        self.mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)
        self.X = tf.placeholder(tf.float32, [None, 784]) # input image == 28x28 pixels == 784
        self.Y = tf.placeholder(tf.float32, [None, 10])  # one hot vector of output (0~9)

    def init_session(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def build_model(self):
        W = tf.Variable(tf.zeros([784, 10]))        # weight
        b = tf.Variable(tf.zeros([10]))             # bias
        self.model = tf.add(tf.matmul(self.X, W), b, name='model') # predicted model

    def build_training_op(self):
        # define an optimizer to reduce the loss
        pred_Y = tf.nn.softmax(self.model)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(pred_Y), reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

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
        for i in range(1000):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            feed_dict = {self.X: batch_xs, self.Y: batch_ys}
            _, loss_val = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            if i % 10 == 0:
                print("step %3d: loss = %.4f" % (i, loss_val))

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        feed_dict = {self.X: self.mnist.test.images, self.Y: self.mnist.test.labels}
        print("accuracy:", self.sess.run(accuracy, feed_dict=feed_dict))

    def infer(self, image):
        feed_dict = {self.X: image.reshape(-1, 784)}
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
