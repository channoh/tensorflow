#!/usr/bin/env python3

#
# https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners
#

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)
    
    X = tf.placeholder(tf.float32, [None, 784]) # input image == 28x28 pixels == 784
    Y = tf.placeholder(tf.float32, [None, 10])  # one hot vector of output (0~9)
    W = tf.Variable(tf.zeros([784, 10]))        # weight
    b = tf.Variable(tf.zeros([10]))             # bias
    pred_Y = tf.nn.softmax(tf.matmul(X, W) + b) # predicted model

    # define an optimizer to reduce the loss
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred_Y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # execute the optimizer
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # train
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, loss_val = sess.run([optimizer, loss], feed_dict={X: batch_xs, Y: batch_ys})
            if i % 10 == 0:
                print("step %3d: loss = %.4f" % (i, loss_val))

        # evaluate
        correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_val = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                     Y: mnist.test.labels})
        print("accuracy:", accuracy_val)

if __name__ == "__main__":
    main()
