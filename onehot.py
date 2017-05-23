import tensorflow as tf


sess = tf.InteractiveSession()

'''
assume batch_size is N, NUM_CLASSES is C
'''
batch_size = 3
NUM_CLASSES = 10


labels = tf.constant([1, 5, 7])
#labels = tf.reshape(labels, [-1, 1])
labels = tf.expand_dims(labels, 1)
labels = tf.Print(labels, [labels])

indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
#indices = tf.Print(indices, [indices])i
concated = tf.concat(1, [indices, labels])
concated = tf.Print(concated, [concated])
'''
tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)
'''
onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
onehot_labels = tf.Print(onehot_labels, [onehot_labels])

with tf.Session() as sess:
    result=sess.run(onehot_labels)
    print(labels)
    print(indices)
    print(concated)
    print(result)
