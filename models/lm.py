import os
import codecs
import numpy as np
import tensorflow as tf



def make_custom_embedding_matrix(vocab, hps):
    if os.path.exists(hps.custom_embed_path + '.npy'):
        mat = np.load(hps.custom_embed_path + '.npy')
        return tf.Variable(mat, trainable=True, name='embed_matrix')

    emb_mat = load_pretrain_word_embedding_matrix(hps, vocab)
    emb_mat = np.array(emb_mat, dtype=np.float32)
    np.save(hps.custom_embed_path, emb_mat)
    assert os.path.exists(hps.custom_embed_path + '.npy')

    emb_mat = tf.Variable(emb_mat, trainable=True, name='embed_matrix')

    return emb_mat


def load_pretrain_word_embedding_matrix(hps, vocab):
    """
    Return the pretrained glove matrix
    """
    assert os.path.exists(hps.embed_path)
    words = {k:None for k in vocab.words}

    with codecs.getreader('utf-8')(tf.gfile.GFile(hps.embed_path, 'rb')) as f:
        for line in f:
            line = line.strip().split()
            if line[0] in vocab.words:
                vec = [float(var) for var in line[1:]]
                assert len(vec) == hps.embed_dim
                words[line[0]] = vec

    unk_cnt = 0
    for word, vec in words.items():
        if vec is None:
            vec = np.random.normal(size=hps.embed_dim)
            words[word] = vec
            unk_cnt += 1

    print("UNK is {}".format(unk_cnt))
    assert len(list(words.keys())) == len(vocab.words)
    matrix = []
    for word in vocab.words:
        matrix.append(words[word])
    return matrix


class LMModel:
    def __init__(self, vocab, hps):
        self.hps = hps
        self.vocab = vocab

        self.dec_out_states, self.dec_outputs, self.attn_dists = None, None, None
        self.add_placeholder()

        self.build_model()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.hps.mode in ['train', 'lm_train']:
            self.add_train_op()

        self.saver = tf.train.Saver(max_to_keep=10)
        self.summaries = tf.summary.merge_all()

    def build_model(self):
        print('model build start')
        with tf.variable_scope('model'):
            self.rand_unif_init = tf.random_uniform_initializer(-self.hps.rand_unif_init_size, self.hps.rand_unif_init_size, seed=777)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.hps.trunc_norm_init_std)

            with tf.variable_scope('embedding'):
                print('add embedding')
                self.add_embedding()

            with tf.variable_scope('encoder'):
                print("add encoding")
                self.add_encoder()

            with tf.variable_scope('decoder'):
                print('add decoding')
                self.add_decoder()

            with tf.variable_scope('output_projection'):
                self.add_output_projection()

            self.add_loss_calculation()

    def add_embedding(self):
        embedding_matrix = make_custom_embedding_matrix(self.vocab, self.hps)
        self.emb_text_inp = tf.nn.embedding_lookup(embedding_matrix, self.text_inp_batch)

    def add_encoder(self):
        cell_fw1 = tf.nn.rnn_cell.GRUCell(self.hps.hidden_dim, kernel_initializer=self.rand_unif_init)
        cell_bw1 = tf.nn.rnn_cell.GRUCell(self.hps.hidden_dim, kernel_initializer=self.rand_unif_init)

        cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=self.hps.keep_rate)
        cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw1, input_keep_prob=self.hps.keep_rate)

        encoder_outputs, (self.bi_fw_st, self.bi_bw_st) = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, self.emb_text_inp, dtype=tf.float32,
                                                                              sequence_length=self.text_lens, swap_memory=True)
        # encoder_outputs: rnn-output of each time step.
        # shape: [2(bi-directional), batch_size, max_enc_len(dynamic), hidden_dim]
        #
        # bi_fw_st (or bi_bw_st): output state of forward (or backward) rnn cell.
        # shape: [batch_size, hidden_dim]

        self.encoder_outputs = tf.concat(axis=2, values=encoder_outputs)

    def reduce_state(self):
        """
        Reduce the bi-directional encoder output state size.
        """
        w_reduce = tf.get_variable('w_reduce', [self.hps.hidden_dim * 2, self.hps.decoder_hidden_dim],
                                   dtype=tf.float32, initializer=self.trunc_norm_init)
        b_reduce = tf.get_variable('b_reduce', [self.hps.decoder_hidden_dim],
                                   dtype=tf.float32, initializer=self.trunc_norm_init)
        concat_h = tf.concat(axis=1, values=[self.bi_bw_st, self.bi_fw_st])

        new_h = tf.nn.relu(tf.matmul(concat_h, w_reduce) + b_reduce)
        self.dec_init_state = new_h

    def add_decoder(self):
        cell_fw1 = tf.nn.rnn_cell.GRUCell(self.hps.decoder_hidden_dim, kernel_initializer=self.rand_unif_init)
        cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=self.hps.keep_rate)

        self.dec_outputs, state = tf.nn.dynamic_rnn(cell_fw1, inputs=self.emb_text_inp, dtype=tf.float32)

    def add_output_projection(self):
        with tf.variable_scope('dec_output_projection'):
            w = tf.get_variable('w1', [self.hps.decoder_hidden_dim, len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)
            v = tf.get_variable('v1', [len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)

            dec_flatten = tf.reshape(
                tf.stack(self.dec_outputs), [-1, self.hps.decoder_hidden_dim])

            dec_vocab_scores = tf.nn.xw_plus_b(dec_flatten, w, v)
            dec_vocab_scores = tf.reshape(dec_vocab_scores,
                                          [-1, self.hps.batch_size, len(self.vocab.words)])
            dec_vocab_dists = tf.nn.softmax(dec_vocab_scores + 1e-12)

            self.dec_vocab_scores = tf.unstack(dec_vocab_scores)
            self.dec_vocab_dists = tf.unstack(dec_vocab_dists)

            """
            Encoder output projection.
            """
            w2 = tf.get_variable('w2', [2 * self.hps.hidden_dim, len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)
            v2 = tf.get_variable('v2', [len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)

            enc_flatten = tf.reshape(tf.stack(self.encoder_outputs), [-1, 2 * self.hps.hidden_dim])

            enc_vocab_scores = tf.nn.xw_plus_b(enc_flatten, w2, v2)
            enc_vocab_scores = tf.reshape(enc_vocab_scores, [-1, self.hps.batch_size, len(self.vocab.words)])
            enc_vocab_dists = tf.nn.softmax(enc_vocab_scores + 1e-12)

            self.enc_vocab_scores = tf.unstack(enc_vocab_scores)
            self.enc_vocab_dists = tf.unstack(enc_vocab_dists)

    def add_train_op(self):
        loss_to_minimize = self.loss

        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self.hps.max_grad_norm)

        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdamOptimizer(self.hps.learning_rate)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def add_loss_calculation(self):
        self.dec_loss = tf.contrib.seq2seq.sequence_loss(
            tf.stack(self.dec_vocab_scores, axis=1),
            self.text_tgt_batch, self.text_pad_masks)
        self.enc_loss = tf.contrib.seq2seq.sequence_loss(
            tf.stack(self.enc_vocab_scores, axis=1),
            self.text_tgt_batch, self.text_pad_masks)

        self.loss = self.enc_loss + self.dec_loss

        tf.summary.scalar('enc_loss', self.enc_loss)
        tf.summary.scalar('dec_loss', self.dec_loss)
        tf.summary.scalar('loss', self.loss)

    def add_placeholder(self):
        self.text_inp_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.max_lm_len], name='text_inp_batch')
        self.text_tgt_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.max_lm_len], name='text_tgt_batch')
        self.text_lens = tf.placeholder(tf.int32, [self.hps.batch_size], name='text_lens')
        self.text_pad_masks = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_lm_len], name='text_pad_masks')

    def run_debug_step(self, batch, sess):
        feeddict = self.make_feeddict(batch)
        to_return = {
            'output':[self.bi_fw_st, self.bi_bw_st]
        }
        return sess.run(to_return, feeddict)

    def run_step(self, batch, sess, is_train=False):
        feeddict = self.make_feeddict(batch)
        to_return = {
            'summaries': self.summaries,
            'loss': self.loss,
            'global_step': self.global_step
        }
        if is_train: to_return['train_op'] = self.train_op
        return sess.run(to_return, feeddict)

    def make_feeddict(self, batch):
        feed_dict = {}
        feed_dict[self.text_inp_batch] = batch.text_inp_batch
        feed_dict[self.text_tgt_batch] = batch.text_tgt_batch
        feed_dict[self.text_lens] = batch.text_lens
        feed_dict[self.text_pad_masks] = batch.text_pad_mask

        return feed_dict
