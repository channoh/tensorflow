#!/usr/bin/env python3

#
# https://www.tensorflow.org/versions/r1.0/get_started/mnist/pros
#

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time

def main():
    mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)
    
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
        start = time.time()
        for i in range(20000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            if i % 100 == 0:
                # evaluate
                accuracy_val = accuracy.eval(feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
                print("step %d: accuracy= %.4f, %.2fs" % (i, accuracy_val, time.time() - start))
                start = time.time()
            optimizer.run(feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})

            # .reshape(-1, 28, 28, 1),
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                Y: mnist.test.labels,
                                                keep_prob: 1.0})
        print("accuracy:", accuracy_val)

if __name__ == "__main__":
    main()
