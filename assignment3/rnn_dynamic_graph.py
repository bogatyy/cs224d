import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tree as tr
from utils import Vocab

RESET_AFTER = 50
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


class RNN_Model():

  def load_data(self):
    """Loads train/dev/test data and builds vocabulary."""
    self.train_data, self.dev_data, self.test_data = tr.simplified_data(
        700, 100, 200)

    # build vocab from training data
    self.vocab = Vocab()
    train_sents = [t.get_words() for t in self.train_data]
    self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

  def inference(self, tree, predict_only_root=False):
    """For a given tree build the RNN models computation graph up to where it
        may be used for inference.
    Args:
        tree: a Tree object on which to build the computation graph for the
          RNN
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    node_tensors = self.add_model(tree.root)
    if predict_only_root:
      node_tensors = node_tensors[tree.root]
    else:
      node_tensors = [tensor for node, tensor in node_tensors.iteritems()
                      if node.label != 2]
      node_tensors = tf.concat(0, node_tensors)
    return self.add_projections(node_tensors)

  def add_model_vars(self):
    '''
    You model contains the following parameters:
        embedding:  tensor(vocab_size, embed_size)
        W1:         tensor(2* embed_size, embed_size)
        b1:         tensor(1, embed_size)
        U:          tensor(embed_size, output_size)
        bs:         tensor(1, output_size)
    Hint: Add the tensorflow variables to the graph here and *reuse* them
    while building
            the compution graphs for composition and projection for each
            tree
    Hint: Use a variable_scope "Composition" for the composition layer, and
          "Projection") for the linear transformations preceding the
          softmax.
    '''
    with tf.variable_scope('Embeddings'):
      tf.get_variable('embeddings', [len(self.vocab), self.config.embed_size])
    with tf.variable_scope('Composition'):
      tf.get_variable('W1',
                      [2 * self.config.embed_size, self.config.embed_size])
      tf.get_variable('b1', [1, self.config.embed_size])
    with tf.variable_scope('Projection'):
      tf.get_variable('U', [self.config.embed_size, self.config.label_size])
      tf.get_variable('bs', [1, self.config.label_size])

  def embed_word(self, word):
    with tf.variable_scope('Embeddings', reuse=True):
      embeddings = tf.get_variable('embeddings')
    with tf.device('/cpu:0'):
      return tf.expand_dims(
          tf.nn.embedding_lookup(embeddings, self.vocab.encode(word)), 0)

  def add_model(self, node):
    """Recursively build the model to compute the phrase embeddings in the tree

    Hint: Refer to tree.py and vocab.py before you start. Refer to
          the model's vocab with self.vocab
    Hint: Reuse the "Composition" variable_scope here
    Hint: Store a node's vector representation in node.tensor so it can be
          used by it's parent
    Hint: If node is a leaf node, it's vector representation is just that of
    the
          word vector (see tf.gather()).
    Args:
        node: a Node object
    Returns:
        node_tensors: Dict: key = Node, value = tensor(1, embed_size)
    """
    with tf.variable_scope('Composition', reuse=True):
      W1 = tf.get_variable('W1')
      b1 = tf.get_variable('b1')

    node_tensors = OrderedDict()
    curr_node_tensor = None
    if node.isLeaf:
      curr_node_tensor = self.embed_word(node.word)
    else:
      node_tensors.update(self.add_model(node.left))
      node_tensors.update(self.add_model(node.right))
      node_input = tf.concat(
          1, [node_tensors[node.left], node_tensors[node.right]])
      curr_node_tensor = tf.nn.relu(tf.matmul(node_input, W1) + b1)

    node_tensors[node] = curr_node_tensor
    return node_tensors

  def add_projections(self, node_tensors):
    """Add projections to the composition vectors to compute the raw sentiment scores

    Hint: Reuse the "Projection" variable_scope here
    Args:
        node_tensors: tensor(?, embed_size)
    Returns:
        output: tensor(?, label_size)
    """
    with tf.variable_scope('Projection', reuse=True):
      U = tf.get_variable('U')
      bs = tf.get_variable('bs')
    logits = tf.matmul(node_tensors, U) + bs
    return logits

  def loss(self, logits, labels):
    """Adds loss ops to the computational graph.

    Hint: Use sparse_softmax_cross_entropy_with_logits
    Hint: Remember to add l2_loss (see tf.nn.l2_loss)
    Args:
        logits: tensor(num_nodes, output_size)
        labels: python list, len = num_nodes
    Returns:
        loss: tensor 0-D
    """
    softmax_loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.constant(
            labels)))
    with tf.variable_scope('Composition', reuse=True):
      W1 = tf.get_variable('W1')
    with tf.variable_scope('Projection', reuse=True):
      U = tf.get_variable('U')
    return softmax_loss + self.config.l2 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(U)
                                           )

  def training(self, loss_tensor):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable
    variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.GradientDescentOptimizer for this model.
            Calling optimizer.minimize() will return a train_op object.

    Args:
        loss: tensor 0-D
    Returns:
        train_op: tensorflow op for training.
    """
    return tf.train.GradientDescentOptimizer(self.config.lr).minimize(
        loss_tensor)

  def predictions(self, y):
    """Returns predictions from sparse scores

    Args:
        y: tensor(?, label_size)
    Returns:
        predictions: tensor(?)
    """
    return tf.argmax(y, 1)

  def __init__(self, config):
    self.config = config
    self.load_data()

  def predict(self, trees, weights_path, get_loss=False):
    """Make predictions from the provided model."""
    results = []
    losses = []
    for i in xrange(int(math.ceil(len(trees) / float(RESET_AFTER)))):
      with tf.Graph().as_default(), tf.Session() as sess:
        self.add_model_vars()
        saver = tf.train.Saver()
        saver.restore(sess, weights_path)
        for tree in trees[i * RESET_AFTER:(i + 1) * RESET_AFTER]:
          logits = self.inference(tree, True)
          predictions = self.predictions(logits)
          root_prediction = sess.run(predictions)[0]
          if get_loss:
            root_label = tree.root.label
            loss = sess.run(self.loss(logits, [root_label]))
            losses.append(loss)
          results.append(root_prediction)
    return results, losses

  def run_epoch(self, new_model=False, verbose=True):
    step = 0
    loss_history = []
    random.shuffle(self.train_data)
    while step < len(self.train_data):
      with tf.Graph().as_default(), tf.Session() as sess:
        self.add_model_vars()
        if new_model:
          init = tf.initialize_all_variables()
          sess.run(init)
          new_model = False
        else:
          saver = tf.train.Saver()
          saver.restore(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
        for _ in xrange(RESET_AFTER):
          if step >= len(self.train_data):
            break
          tree = self.train_data[step]
          logits = self.inference(tree)
          labels = [l for l in tree.labels if l != 2]
          loss_tensor = self.loss(logits, labels)
          train_op = self.training(loss_tensor)
          loss_value, _ = sess.run([loss_tensor, train_op])
          loss_history.append(loss_value)
          if verbose:
            sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
                self.train_data), np.mean(loss_history)))
            sys.stdout.flush()
          step += 1
        saver = tf.train.Saver()
        if not os.path.exists(SAVE_DIR):
          os.makedirs(SAVE_DIR)
        saver.save(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
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

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to
  1
  so you can rapidly iterate.
  """
  config = Config()
  model = RNN_Model(config)
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

  print 'Test'
  print '=-=-='
  predictions, _ = model.predict(model.test_data,
                                 SAVE_DIR + '%s.temp' % model.config.model_name)
  labels = [t.root.label for t in model.test_data]
  print model.make_conf(labels, predictions)
  test_acc = np.equal(predictions, labels).mean()
  print 'Test acc: {}'.format(test_acc)
  print 'Time to run inference on dev+test: {}'.format(time.time() - start_time)


if __name__ == '__main__':
  test_RNN()
