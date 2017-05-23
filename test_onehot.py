import tensorflow as tf

num_labels = 10

label_batch = tf.constant([1, 4, 6])
sparse_labels = tf.reshape(label_batch, [-1, 1])
derived_size = tf.shape(label_batch)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat([indices, sparse_labels], 1)
outshape = tf.stack([derived_size, num_labels])
labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

with tf.Session() as sess:
    result = sess.run(labels)
    print(result)
