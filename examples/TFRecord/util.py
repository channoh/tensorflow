
import tensorflow as tf
from PIL import Image
import sys
import os
import glob
import cv2
import numpy as np


def annotate(img_path, x, y, height, width, out_path='annotated.jpg'):
    img_raw = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    img_raw = annotate_raw(img_raw, x, y, height, width, out_path)
    cv2.imwrite(out_path, img_raw)

def annotate_raw(img_raw, x, y, height, width):
    overlay = img_raw.copy()
    opacity = 0.4
    cv2.rectangle(overlay, (x, y), (x+height, y+width), (0,255,0), -1)
    cv2.addWeighted(overlay, opacity, img_raw, 1-opacity, 0, img_raw)
    return img_raw

def read_image_cv2(img_file):
    img = cv2.imread(img_file)
    h, w, _ = img.shape
    # img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8).tostring(), w, h

def read_image_pil(img_file):
    img = Image.open(img_file)
    w, h = img.size
    return np.array(img).tostring(), w, h


def write_to_tfrecord(tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for i in range(6,8):
        train_files = glob.glob('./data/train/base{}/*.jpg'.format(i))
        print("base{}: {}".format(i, len(train_files)))
        for fname in train_files:
            tf_record = create_tfrecord(fname, i)
            writer.write(tf_record.SerializeToString())
    writer.close()


def create_tfrecord(fname, base):
    filename = str.encode(os.path.basename(fname))
    base_text = str.encode("base{}".format(base))
    image_format = b'jpg'
    image_raw, width, height = read_image_pil(fname)
    # image_raw, width, height = read_image_cv2(fname)

    xmins = [166.0]
    xmaxs = [238.0]
    ymins = [146.0]
    ymaxs = [200.0]
    classes = [base+1]

    xmins = [x / float(width) for x in xmins]
    xmaxs = [x / float(width) for x in xmaxs]
    ymins = [y / float(height) for y in ymins]
    ymaxs = [y / float(height) for y in ymaxs]

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image': _bytes_feature(image_raw),
        'bbox/xmins': _float_list_feature(xmins),
        'bbox/xmaxs': _float_list_feature(xmaxs),
        'bbox/ymins': _float_list_feature(ymins),
        'bbox/ymaxs': _float_list_feature(ymaxs),
        'class/label': _int64_list_feature(classes)
        }))
    return tf_record

def read_from_tfrecord(tfrecord_filename):

    filename_queue = tf.train.string_input_producer([tfrecord_filename], name='queue')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'height':        tf.FixedLenFeature([], tf.int64),
        'width':         tf.FixedLenFeature([], tf.int64),
        'image':         tf.FixedLenFeature([], tf.string),
        'bbox/xmins':    tf.VarLenFeature(tf.float32),
        'bbox/xmaxs':    tf.VarLenFeature(tf.float32),
        'bbox/ymins':    tf.VarLenFeature(tf.float32),
        'bbox/ymaxs':    tf.VarLenFeature(tf.float32),
        'class/label':   tf.VarLenFeature(tf.int64)
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [height, width, 3])

    bbox_xmins = tf.sparse_tensor_to_dense(features['bbox/xmins'], default_value=0.0)
    bbox_xmaxs = tf.sparse_tensor_to_dense(features['bbox/xmaxs'], default_value=0.0)
    bbox_ymins = tf.sparse_tensor_to_dense(features['bbox/ymins'], default_value=0.0)
    bbox_ymaxs = tf.sparse_tensor_to_dense(features['bbox/ymaxs'], default_value=0.0)

    label = tf.sparse_tensor_to_dense(features['class/label'], default_value=0)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        cnt = 0
        for _ in tf.python_io.tf_record_iterator(tfrecord_filename):
            h, w, image, bxmin, bxmax, bymin, bymax, l = sess.run([height, width, img, bbox_xmins, bbox_xmaxs, bbox_ymins, bbox_ymaxs, label])
            # print(h, w, bxmin[0], bxmax[0], bymin[0], bymax[0], l)
            cnt += 1

        coord.request_stop()
        coord.join(threads)
        print(cnt)

        ## draw image from tfrecord
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        x0 = int(w * bxmin[0])
        y0 = int(h * bymin[0])
        x2 = int(w * bxmax[0])
        y2 = int(h * bymax[0])
        image = annotate_raw(image, x0, y0, x2 - x0, y2 - y0)
        cv2.imwrite('test.jpg', image)


if __name__ == "__main__":
    # annotate("./data/train/base7/mlb04-00001499.jpg", 166, 146, 72, 54)

    tfrecord_filename = "./mlb_train.tfrecord"

    ## Writer
    write_to_tfrecord(tfrecord_filename)

    ## Reader
    read_from_tfrecord(tfrecord_filename)
