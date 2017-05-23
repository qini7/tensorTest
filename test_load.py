import os
import tensorflow as tf

def read_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                        features = {
                                            'label' : tf.FixedLenFeature([], tf.int64),
                                            #'height' : tf.FixedLenFeature([], tf.int64),
                                            #'width' : tf.FixedLenFeature([], tf.int64),
                                            'img_raw' : tf.FixedLenFeature([], tf.string),
                                        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    #h = tf.cast(features['height'], tf.int64)
    #w = tf.cast(features['width'], tf.int64)
    img = tf.reshape(img, [1024])
    img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)

    return img, label


img_test, label_test = read_decode("test.tfrecords")

img_test_batch, label_test_batch = tf.train.shuffle_batch([img_test, label_test],
                                                        batch_size = 300, capacity = 1000,
                                                        min_after_dequeue = 900)

# net
x = tf.placeholder(tf.float32, [None, 1024])
W = tf.Variable(tf.zeros([1024, 10]), name = 'W')
b = tf.Variable(tf.zeros([10]), name = 'b')
y = tf.matmul(x, W) + b

batch_size = tf.placeholder(tf.int64)

#ground truth
y_pre = tf.placeholder(tf.int32, [None])
labels = tf.reshape(y_pre, [-1, 1])
derived_size = tf.shape(y_pre)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat([indices, labels], 1)
outshape = tf.stack([derived_size, 10])
y_ = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


saver = tf.train.Saver({'W': W, 'b' : b})
saver.restore(sess, 'E:/githubLib/tensorTest/digit_data/1.model')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

try:
    #test code
    val, l = sess.run([img_test_batch, label_test_batch])
    print(accuracy.eval(feed_dict = {x : val, y_pre : l, batch_size : 300}))

except tf.errors.OutOfRangeError:
    print("stop")
finally:
    coord.request_stop()
coord.join(threads)

sess.close()
