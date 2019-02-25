import tensorflow as tf
import os
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v3 = tf.get_variable("v3", shape=[3], initializer=tf.ones_initializer)
v4 = v1 + v3

# Add ops to save and restore all the variables.
saver = tf.train.Saver({"v1":v1})

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Restore variables from disk.
  saver.restore(sess, os.path.join(os.getcwd(), "/tmp/model.ckpt"))
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v3 : %s" % v3.eval())
  print("v2 : %s" % v4.eval())