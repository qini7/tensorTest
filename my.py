import tensorflow as tf

with tf.Graph().as_default() as g:
    fib_matrix = tf.constant([[0.0, 1.0],
                              [1.0, 1.0]])

    x = tf.Variable([[1.0, 1.0],
                    [1.0, 1.0]], name = 'x')

    next_fib = tf.matmul(x, fib_matrix)

    assign_op = tf.assign(x, next_fib)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(10):
            sess.run(assign_op)
            print(sess.run(x))
