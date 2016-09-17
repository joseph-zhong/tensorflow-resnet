# from convert import checkpoint_fn, meta_fn
import os, time
from utils import print_prob, load_image, load_cv_img
import tensorflow as tf

layers = 50

img = load_cv_img("data/cat.jpg")

sess = tf.Session()

new_saver = tf.train.import_meta_graph(os.path.expandvars('$HOME/ws/var/shared/local/data/model_checkpoints/resnet/152/ResNet-L152.meta'))
new_saver.restore(sess, os.path.expandvars('$HOME/ws/var/shared/local/data/model_checkpoints/resnet/152/ResNet-L152.ckpt'))

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
# for op in graph.get_operations():
#     print op.name

# init = tf.initialize_all_variables()
# sess.run(init)
print "graph restored"

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}
start_time = time.time()
prob = sess.run(prob_tensor, feed_dict=feed_dict)
print time.time() - start_time

print_prob(prob[0])
