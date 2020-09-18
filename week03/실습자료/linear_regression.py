import tensorflow as tf

# Model Parameter
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_mean(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

MaxIter = 10

# training loop
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for i in range(MaxIter):
		curr_W, curr_b, curr_loss = sess.run([W, b, loss], feed_dict={x:x_train, y:y_train})
		print(i, curr_W, curr_b, curr_loss)
		sess.run(train, feed_dict={x:x_train, y:y_train})
	curr_W, curr_b, curr_loss = sess.run([W, b, loss], feed_dict={x:x_train, y:y_train})
	print(i+1, curr_W, curr_b, curr_loss)