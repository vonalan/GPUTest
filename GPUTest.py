import itertools 

import numpy as np 
import tensorflow as tf

def main(): 
    x = tf.placeholder(tf.float32, shape=[None,4096])
    t = tf.placeholder(tf.float32, shape=[None,1])
    w = tf.get_variable('weights', shape=[4096,1])
    b = tf.get_variable('bias', shape=[1])
    y = tf.add(tf.matmul(x, w), b)
    
    loss = tf.reduce_mean(tf.square(t-y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in itertools.count(): 
        xs = np.random.random((10,4096))
        ts = np.random.random((10,1))
        cost, _ = sess.run([loss, optimizer], feed_dict={x: xs, t: ts})
        print('epoch: %d, cost: %f'%(i+1, cost))
    sess.close()

if __name__ == '__main__': 
   with tf.device('/gpu:0'): 
       main()
