from model import *
config = CONFIG()
with tf.Session() as sess:
    model = RNET()
    model.train(sess=sess)
