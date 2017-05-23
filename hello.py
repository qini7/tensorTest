import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

linear_model = W * x + b

loss = tf.reduce_sum(tf.square(linear_model - y_))

#initialize
init = tf.global_variables_initializer()

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x:x_train, y_:y_train})

    #evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y_:y_train})
    #print(sess.run([W,b]))
    print(curr_W, curr_b, curr_loss)
