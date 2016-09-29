import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils
import tree

MODEL_STR = 'rnn_embed=%d_l2=%f_lr=%f.weights'
SAVE_DIR = './weights/'


class Config(object):
  """Holds model hyperparams and data information.
  Model objects are passed a Config() object at instantiation.
  """
  embed_size = 35
  label_size = 2
  early_stopping = 2
  anneal_threshold = 0.99
  anneal_by = 1.5
  max_epochs = 30
  lr = 0.01
  l2 = 0.02

  model_name = MODEL_STR % (embed_size, l2, lr)


class RecursiveNetStaticGraph():

  def __init__(self, config):
    self.config = config

    # Load train data and build vocabulary
    self.train_data, self.dev_data, self.test_data = tree.simplified_data(700,
                                                                          100,
                                                                          200)
    self.vocab = utils.Vocab()
    train_sents = [t.get_words() for t in self.train_data]
    self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    # add input placeholders
    self.is_leaf_placeholder = tf.placeholder(
        tf.bool, (None), name='is_leaf_placeholder')
    self.left_children_placeholder = tf.placeholder(
        tf.int32, (None), name='left_children_placeholder')
    self.right_children_placeholder = tf.placeholder(
        tf.int32, (None), name='right_children_placeholder')
    self.node_word_indices_placeholder = tf.placeholder(
        tf.int32, (None), name='node_word_indices_placeholder')
    self.labels_placeholder = tf.placeholder(
        tf.int32, (None), name='labels_placeholder')

    # add model variables
    with tf.variable_scope('Embeddings'):
      embeddings = tf.get_variable('embeddings',
                                   [len(self.vocab), self.config.embed_size])
    with tf.variable_scope('Composition'):
      W1 = tf.get_variable('W1',
                           [2 * self.config.embed_size, self.config.embed_size])
      b1 = tf.get_variable('b1', [1, self.config.embed_size])
    with tf.variable_scope('Projection'):
      U = tf.get_variable('U', [self.config.embed_size, self.config.label_size])
      bs = tf.get_variable('bs', [1, self.config.label_size])

    # build recursive graph

    tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

    def embed_word(word_index):
      with tf.device('/cpu:0'):
        return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(left_tensor, right_tensor):
      return tf.nn.relu(tf.matmul(tf.concat(1, [left_tensor, right_tensor]), W1) + b1)

    def loop_body(tensor_array, i):
      node_is_leaf = tf.gather(self.is_leaf_placeholder, i)
      node_word_index = tf.gather(self.node_word_indices_placeholder, i)
      left_child = tf.gather(self.left_children_placeholder, i)
      right_child = tf.gather(self.right_children_placeholder, i)
      node_tensor = tf.cond(
          node_is_leaf,
          lambda: embed_word(node_word_index),
          lambda: combine_children(tensor_array.read(left_child),
                                   tensor_array.read(right_child)))
      tensor_array = tensor_array.write(i, node_tensor)
      i = tf.add(i, 1)
      return tensor_array, i

    loop_cond = lambda tensor_array, i: \
        tf.less(i, tf.squeeze(tf.shape(self.is_leaf_placeholder)))
    self.tensor_array, _ = tf.while_loop(
        loop_cond, loop_body, [tensor_array, 0], parallel_iterations=1)

    # add projection layer
    self.logits = tf.matmul(self.tensor_array.concat(), U) + bs
    self.root_logits = tf.matmul(
        self.tensor_array.read(self.tensor_array.size() - 1), U) + bs
    self.root_prediction = tf.squeeze(tf.argmax(self.root_logits, 1))

    # add loss layer
    regularization_loss = self.config.l2 * (
        tf.nn.l2_loss(W1) + tf.nn.l2_loss(U))
    included_indices = tf.where(tf.less(self.labels_placeholder, 2))
    self.full_loss = regularization_loss + tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.gather(self.logits, included_indices), tf.gather(
                self.labels_placeholder, included_indices)))
    self.root_loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.root_logits, self.labels_placeholder[-1:]))

    # add training op
    self.train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(
        self.full_loss)

  def build_feed_dict(self, node):
    nodes_list = []
    tree.leftTraverse(node, lambda node, args: args.append(node), nodes_list)
    node_to_index = OrderedDict()
    for i in xrange(len(nodes_list)):
      node_to_index[nodes_list[i]] = i
    feed_dict = {
        self.is_leaf_placeholder: [node.isLeaf for node in nodes_list],
        self.left_children_placeholder: [node_to_index[node.left] if
                                         not node.isLeaf else -1
                                         for node in nodes_list],
        self.right_children_placeholder: [node_to_index[node.right] if
                                          not node.isLeaf else -1
                                          for node in nodes_list],
        self.node_word_indices_placeholder: [self.vocab.encode(node.word) if
                                             node.word else -1
                                             for node in nodes_list],
        self.labels_placeholder: [node.label for node in nodes_list]
    }
    return feed_dict

  def predict(self, trees, weights_path, get_loss=False):
    """Make predictions from the provided model."""
    results = []
    losses = []
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess, weights_path)
      for tree in trees:
        feed_dict = self.build_feed_dict(tree.root)
        if get_loss:
          root_prediction, loss = sess.run(
              [self.root_prediction, self.root_loss], feed_dict=feed_dict)
          losses.append(loss)
        else:
          root_prediction = sess.run(self.root_prediction, feed_dict=feed_dict)
        results.append(root_prediction)
    return results, losses

  def run_epoch(self, new_model=False, verbose=True):
    loss_history = []
    # training
    random.shuffle(self.train_data)
    with tf.Session() as sess:
      if new_model:
        sess.run(tf.initialize_all_variables())
      else:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
      for step, tree in enumerate(self.train_data):
        feed_dict = self.build_feed_dict(tree.root)
        loss_value, _ = sess.run([self.full_loss, self.train_op],
                                 feed_dict=feed_dict)
        loss_history.append(loss_value)
        if verbose:
          sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
              self.train_data), np.mean(loss_history)))
          sys.stdout.flush()
      saver = tf.train.Saver()
      if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
      saver.save(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
    # statistics
    train_preds, _ = self.predict(self.train_data,
                                  SAVE_DIR + '%s.temp' % self.config.model_name)
    val_preds, val_losses = self.predict(
        self.dev_data,
        SAVE_DIR + '%s.temp' % self.config.model_name,
        get_loss=True)
    train_labels = [t.root.label for t in self.train_data]
    val_labels = [t.root.label for t in self.dev_data]
    train_acc = np.equal(train_preds, train_labels).mean()
    val_acc = np.equal(val_preds, val_labels).mean()

    print
    print 'Training acc (only root node): {}'.format(train_acc)
    print 'Valiation acc (only root node): {}'.format(val_acc)
    print self.make_conf(train_labels, train_preds)
    print self.make_conf(val_labels, val_preds)
    return train_acc, val_acc, loss_history, np.mean(val_losses)

  def train(self, verbose=True):
    complete_loss_history = []
    train_acc_history = []
    val_acc_history = []
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_epoch = 0
    stopped = -1
    for epoch in xrange(self.config.max_epochs):
      print 'epoch %d' % epoch
      if epoch == 0:
        train_acc, val_acc, loss_history, val_loss = self.run_epoch(
            new_model=True)
      else:
        train_acc, val_acc, loss_history, val_loss = self.run_epoch()
      complete_loss_history.extend(loss_history)
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      #lr annealing
      epoch_loss = np.mean(loss_history)
      if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
        self.config.lr /= self.config.anneal_by
        print 'annealed lr to %f' % self.config.lr
      prev_epoch_loss = epoch_loss

      #save if model has improved on val
      if val_loss < best_val_loss:
        shutil.copyfile(SAVE_DIR + '%s.temp' % self.config.model_name,
                        SAVE_DIR + '%s' % self.config.model_name)
        best_val_loss = val_loss
        best_val_epoch = epoch

      # if model has not imprvoved for a while stop
      if epoch - best_val_epoch > self.config.early_stopping:
        stopped = epoch
        #break
    if verbose:
      sys.stdout.write('\r')
      sys.stdout.flush()

    print '\n\nstopped at %d\n' % stopped
    return {
        'loss_history': complete_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }

  def make_conf(self, labels, predictions):
    confmat = np.zeros([2, 2])
    for l, p in itertools.izip(labels, predictions):
      confmat[l, p] += 1
    return confmat


def plot_loss_history(stats):
  plt.plot(stats['loss_history'])
  plt.title('Loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.savefig('loss_history.png')
  plt.show()


def test_RNN():
  """Test RNN model implementation.
  """
  config = Config()
  model = RecursiveNetStaticGraph(config)
  #graph_def = tf.get_default_graph().as_graph_def()
  #with open('static_graph.pb', 'wb') as f:
  #  f.write(graph_def.SerializeToString())

  start_time = time.time()
  stats = model.train(verbose=True)
  print 'Training time: {}'.format(time.time() - start_time)

  plot_loss_history(stats)

  start_time = time.time()
  val_preds, val_losses = model.predict(
      model.dev_data,
      SAVE_DIR + '%s.temp' % model.config.model_name,
      get_loss=True)
  val_labels = [t.root.label for t in model.dev_data]
  val_acc = np.equal(val_preds, val_labels).mean()
  print val_acc

  print '-' * 20
  print 'Test'
  predictions, _ = model.predict(model.test_data,
                                 SAVE_DIR + '%s.temp' % model.config.model_name)
  labels = [t.root.label for t in model.test_data]
  print model.make_conf(labels, predictions)
  test_acc = np.equal(predictions, labels).mean()
  print 'Test acc: {}'.format(test_acc)
  print 'Inference time, dev+test: {}'.format(time.time() - start_time)
  print '-' * 20


if __name__ == '__main__':
  test_RNN()
