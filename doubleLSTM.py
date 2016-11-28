import tensorflow as tf
import tsModule as tmod
import numpy as np
import matplotlib.pyplot as plt

n_periods = 30 #13
n_lstm    = 200 #200
n_hidden1  = 100 #100
n_hidden2  = 50 #50
learning_rate = 0.005 #0.005
n_epoch = 400 #400
n_batches = 50 #50

# loads data
data = tmod.OzoneData()

# placeholders for input and target
X = tf.placeholder(tf.float32, shape = [None, n_periods, 1])
Y = tf.placeholder(tf.float32, shape = [None, 1])

# unroll ltsm and feed into cells
temp = tf.unpack(X, axis = 1)
lstm = tf.nn.rnn_cell.BasicLSTMCell(n_lstm, forget_bias=1.0, activation=tf.nn.relu)
lstm_outputs, states = tf.nn.rnn(lstm, temp, dtype = tf.float32)

# map from ltsm layer to hidden layer 1
with tf.variable_scope('hidden1'):
   W = tf.Variable(tf.random_normal([n_lstm, n_hidden1]))
   b = tf.Variable(tf.random_normal([n_hidden1]))
   
   hidden1_outputs = tf.nn.relu(tf.matmul(lstm_outputs[-1], W) + b)
   
   L2hidden1 = tf.nn.l2_loss(W)

# map from hidden layer 1 to hidden layer 2
with tf.variable_scope('hidden2'):
   W = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
   b = tf.Variable(tf.random_normal([n_hidden2]))
   
   hidden2_outputs = tf.matmul(hidden1_outputs, W) + b
   
   L2hidden2 = tf.nn.l2_loss(W)

with tf.variable_scope('out'):
   W = tf.Variable(tf.random_normal([n_hidden2, 1]))
   b = tf.Variable(tf.random_normal([1]))
   
   pred = tf.matmul(hidden2_outputs, W) + b
   
   L2out = tf.nn.l2_loss(W)

cost = tf.reduce_mean(tf.square(pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as sess:
   sess.run(init)
   
   error_curve = np.zeros(n_epoch)
   
   for i in range(n_epoch):
      if i % 100 == 0:
         print('\nTraining epoch', i)
      input, target = data.random_batch(n_batches, n_periods)

      error_curve[i] = sess.run( cost, feed_dict = {X: np.reshape(input, [n_batches, n_periods, 1]), Y: target})
      
      sess.run( optimizer, feed_dict = {X: np.reshape(input, [n_batches, n_periods, 1]), Y: target})
   
   one_day_prediction = np.zeros(data.n_obs)
   for i in range(n_periods, data.n_obs):
      raw_input =np.reshape(data.get_input(i-n_periods, n_periods), [1, n_periods, 1])
      one_day_prediction[i] = sess.run(pred, feed_dict = {X: raw_input})

fig = plt.figure()

fig.suptitle('period:{0}, LSTM:{1}, H1:{2}, H2:{3}, {4}, n_epoch:{5}, {6}'.format(n_periods, n_lstm, n_hidden1, n_hidden2, learning_rate, n_epoch, n_batches))
plt.subplot(3,1,1)
plt.plot(one_day_prediction[n_periods:-1], 'r', label = 'pre')
plt.plot(data.y[n_periods:-1], 'b', label = 'ac')
plt.legend()

plt.subplot(3,1,2)
plt.acorr((one_day_prediction-data.y)[n_periods:-1], maxlags=26, label = 'auto')
plt.legend()

plt.subplot(3,1,3)
plt.plot(error_curve, label = 'err')
plt.axis([0, n_epoch, 0, 1])
plt.legend()


plt.show()

