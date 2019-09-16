import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def jaccard_distance(text1, text2):
    """ Measure the jaccard distance of two different text.
    ARGS:
        text1,2: list of tokens
    RETURN:
        score(float): distance between two text
        """
    intersection = set(text1).intersection(set(text2))
    union = set(text1).union(set(text2))
    return 1 - len(intersection) / len(union)


def load_ckpt(args, saver, sess, ckpt_dir='train', ckpt_id=None):
    while True:
        if ckpt_id is None or ckpt_id == -1:
            ckpt_dir = os.path.join(args.model_path, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=None)
            print(ckpt_dir)
            ckpt_path = ckpt_state.model_checkpoint_path
            print("CKPT_PATH: {}".format(ckpt_path))
            # print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=False)
            saver.restore(sess, ckpt_path)
            return ckpt_path


def gpu_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def assign_specific_gpu(gpu_nums='-1'):
    assert gpu_nums is not None and gpu_nums != '-1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_nums


class Vocab():
    def __init__(self, path='data/vocab.txt'):
        self.word2id, self.id2word = {}, {}
        self.vocabpath = path
        self.read_voca()

    def read_voca(self):
        assert os.path.exists(self.vocabpath)
        with open(self.vocabpath, 'r', encoding='utf8') as f:
            ls = [line.strip() for line in f.readlines()]
        for idx, word in enumerate(ls):
            self.word2id[word] = idx
            self.id2word[idx] = word
        self.unk_id = self.word2id['<UNK>']
        self.beg_id = self.word2id['<BEG>']
        self.eos_id = self.word2id['<EOS>']
        self.pad_id = self.word2id['<PAD>']
        self.words = list(self.word2id.keys())
        self.word_sorted = ls

    def text2ids(self, toks):
        assert isinstance(toks, list) and all([isinstance(tok, str) for tok in toks])
        ids = [self.word2id[tok] if tok in self.word2id else self.unk_id for tok in toks]
        return ids


def get_pretrain_weights(path):
    """ Load pretrain weights and save """
    with tf.Session(config=gpu_config()) as sess:
        ckpt_name = tf.train.latest_checkpoint(path)
        meta_name = ckpt_name + '.meta'
        save_graph = tf.train.import_meta_graph(meta_name)
        save_graph.restore(sess, ckpt_name)
        var_name_list = [var.name for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        vardict = dict()
        for var in var_name_list:
            vardict[var] = sess.run(tf.get_default_graph().get_tensor_by_name(var))
    tf.reset_default_graph()
    return vardict


def print_config(args):
    print('mode: {}'.format(args.mode))
    print('model: {}'.format(args.model))
    print("use pretrain: {}".format(args.use_pretrain))
    print("Batch size: {}".format(args.batch_size))


def assign_pretrain_weights(pretrain_vardicts):
    assign_op, uninitialized_varlist = [], []
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_op_names = []

    for var in all_variables:
        varname = var.name
        new_model_var = tf.get_default_graph().get_tensor_by_name(varname)
        if 'bw_model' in varname: varname = varname.replace('bw_model', 'models')
        if varname in pretrain_vardicts:
            assign_op.append(tf.assign(new_model_var, pretrain_vardicts[varname]))
            assign_op_names.append(varname)
        else:
            if varname.replace('models/decoder/attention_decoder', 'models/decoder/rnn') in pretrain_vardicts:
                corres_varname = varname.replace('models/decoder/attention_decoder', 'models/decoder/rnn')
                assign_op.append(tf.assign(new_model_var, pretrain_vardicts[corres_varname]))
                assign_op_names.append(varname)
            elif varname.replace('models/tgt_encoder/', 'models/encoder/') in pretrain_vardicts:
                corres_varname = varname.replace('models/tgt_encoder/', 'models/encoder/')
                assign_op.append(tf.assign(new_model_var, pretrain_vardicts[corres_varname]))
                assign_op_names.append(varname)
            elif ('/encoder/' in varname or '/decoder/' in varname) and ('kernel:0' in varname or 'bias:0' in varname):
                raise ValueError("{} should be pretrained.".format(varname))
            else:
                uninitialized_varlist.append(var)

    return assign_op, uninitialized_varlist


def ids2tokens(ids, vocab):
    toks = [vocab.id2word[id_] for id_ in ids if id_ not in [vocab.pad_id, vocab.eos_id]]
    return toks


def sample_gumbel(shape, eps=1e-10):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape, minval=0.0, maxval=1.0)
  tf.stop_gradient(U)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temp):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax(y / temp)


def gumbel_softmax(logits, temp, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temp)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


if __name__ == '__main__':
    logit = tf.Variable([[2,2,3,1,4,1,1,2,1,1]],dtype=tf.float32)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logit = tf.nn.softmax(logit)
        res2 = gumbel_softmax(logit, temp=0.67, hard=False)
        print(sess.run(res2))

