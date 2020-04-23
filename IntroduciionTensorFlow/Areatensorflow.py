import tensorflow as tf
#Aqui intercambiamos el modulo de eager, con el compact v1,
#ya que son versiones antiguas
tf.compat.v1.disable_eager_execution()

A=tf.constant([3], dtype=tf.float64)
B=tf.constant([4],dtype=tf.float64)
C=tf.constant([5],dtype=tf.float64)
S=tf.add(A,tf.add(B,C))/2
Are=tf.sqrt(tf.multiply(S,tf.multiply(S-A,tf.multiply(S-B,S-C))))
#Session con v1
with tf.compat.v1.Session() as sess:
    print(Are.eval())
#Place Holder
a1=tf.compat.v1.placeholder(dtype=tf.float64 , shape=None, name=None)
b1=tf.compat.v1.placeholder(dtype=tf.float64 , shape=None, name=None)
c1=tf.compat.v1.placeholder(dtype=tf.float64 , shape=None, name=None)
S1=tf.add(a1,tf.add(b1,c1))/2
Are2=tf.sqrt(tf.multiply(S1,tf.multiply(S1-a1,tf.multiply(S1-b1,S1-c1))))

with tf.compat.v1.Session() as sess:
    Semi=sess.run(S1, feed_dict={
        a1:[3],
        b1:[4],
        c1:[5]})
    Area=sess.run(Are2, feed_dict={
        a1:[3],
        b1:[4],
        c1:[5],
        S1:Semi})
    print(Area)