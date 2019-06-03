import numpy as np
import tensorflow as tf

import vgg16
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)
batch = batch2

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        for_show = {}
        conv1_1 = sess.run(vgg.conv1_1, feed_dict=feed_dict);for_show['conv1_1'] = conv1_1
        conv1_2 = sess.run(vgg.conv1_2, feed_dict=feed_dict);for_show['conv1_2'] = conv1_2
        conv2_1 = sess.run(vgg.conv2_1, feed_dict=feed_dict);for_show['conv2_1'] = conv2_1
        conv2_2 = sess.run(vgg.conv2_2, feed_dict=feed_dict);for_show['conv2_2'] = conv2_2
        conv3_1 = sess.run(vgg.conv3_1, feed_dict=feed_dict);for_show['conv3_1'] = conv3_1
        conv3_2 = sess.run(vgg.conv3_2, feed_dict=feed_dict);for_show['conv3_2'] = conv3_2
        conv3_3 = sess.run(vgg.conv3_3, feed_dict=feed_dict);for_show['conv3_3'] = conv3_3
        conv4_1 = sess.run(vgg.conv4_1, feed_dict=feed_dict);for_show['conv4_1'] = conv4_1
        conv4_2 = sess.run(vgg.conv4_2, feed_dict=feed_dict);for_show['conv4_2'] = conv4_2
        conv4_3 = sess.run(vgg.conv4_3, feed_dict=feed_dict);for_show['conv4_3'] = conv4_3
        conv5_1 = sess.run(vgg.conv5_1, feed_dict=feed_dict);for_show['conv5_1'] = conv5_1
        conv5_2 = sess.run(vgg.conv5_2, feed_dict=feed_dict);for_show['conv5_2'] = conv5_2
        conv5_3 = sess.run(vgg.conv5_3, feed_dict=feed_dict);for_show['conv5_3'] = conv5_3
        for key in for_show.keys():
            utils.save_feature(for_show[key],key)
