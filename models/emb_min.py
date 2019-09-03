import os
import codecs
import numpy as np
import tensorflow as tf
from models.attention import multihead_attention_decoder
from models.basemodel import BaseModel, make_custom_embedding_matrix
from utils import gumbel_softmax
from models.attention import attention_decoder

class DiverEmbMin(BaseModel):
    def __init__(self, vocab, hps):
        super(DiverEmbMin, self).__init__(vocab, hps)

    def build_model(self):
        print('models build start')
        with tf.variable_scope('models'):
            self.rand_unif_init = tf.random_uniform_initializer(-self.hps.rand_unif_init_size, self.hps.rand_unif_init_size, seed=777)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.hps.trunc_norm_init_std)

            with tf.variable_scope('embedding'):
                print('add embedding')
                self.add_embedding()

            with tf.variable_scope('encoder'):
                print("add encoding")
                self.add_encoder()
                self.reduce_state()
                self.add_mechanism()
                self.dec_init_with_mechanism()

            with tf.variable_scope('decoder'):
                print('add decoding')
                self.add_decoder()

            with tf.variable_scope('output_projection'):
                self.add_output_projection()

            if self.hps.mode not in ['decode']:
                self.regularize_by_attn_dists_timestep()
                self.add_loss_calculation()
            else:
                print('#######', len(self.dec_vocab_dists))
                assert len(self.dec_vocab_dists) == 1
                topk_probs_dec, self.topk_ids = tf.nn.top_k(self.dec_vocab_dists[0], self.hps.batch_size * 2)

                self.topk_log_probs = tf.log(topk_probs_dec)

    def reduce_state(self):
        """
        Reduce the bi-directional encoder output state size.
        """
        w_reduce = tf.get_variable('w_reduce', [self.hps.hidden_dim * 2, self.hps.hidden_dim],
                                   dtype=tf.float32, initializer=self.trunc_norm_init)
        b_reduce = tf.get_variable('b_reduce', [self.hps.hidden_dim],
                                   dtype=tf.float32, initializer=self.trunc_norm_init)
        concat_h = tf.concat(axis=1, values=[self.bi_bw_st, self.bi_fw_st])

        new_h = tf.nn.relu(tf.matmul(concat_h, w_reduce) + b_reduce)

        # [batch_size, hidden_dim]
        self.dec_init_state_small = new_h

    def add_mechanism(self):
        self.matrix = tf.get_variable('mech_mat', [self.hps.matrix_num, self.hps.mechanism_dim], dtype=tf.float32,
                                      initializer=self.rand_unif_init)

        W_enc = tf.get_variable('W_enc', [self.hps.hidden_dim, 2 * self.hps.hidden_dim], dtype=tf.float32,
                                initializer=self.trunc_norm_init)

        enc_feature = tf.matmul(self.dec_init_state_small, W_enc)
        maxout_res = tf.contrib.layers.maxout(enc_feature, self.hps.hidden_dim)  # [batch_size, hidden_dim]

        W_attn = tf.get_variable('W_divert_attn', [self.hps.mechanism_dim, self.hps.hidden_dim], dtype=tf.float32,
                                 initializer=self.trunc_norm_init)
        res1 = tf.matmul(self.matrix, W_attn)

        # [batch_size, mechanism_num]
        self.emb_score = tf.nn.softmax(tf.transpose(tf.matmul(res1, maxout_res, transpose_b=True))) + 1e-12

    def regularize_by_attn_dists_timestep(self):
        # attn_dist shape: [max len, Tensor(BatchSize * matrix_num, dim)]
        attn = tf.stack(self.attn_dists[0])
        attn = tf.reduce_sum(attn, axis=0)  # [B * matrix_num, enc_len]

        attnsum = tf.reduce_sum(attn, axis=1, keep_dims=True)
        attn /= attnsum  # normalized
        attn = tf.reshape(attn, [self.hps.batch_size, self.hps.matrix_num, -1])
        aa_t = tf.matmul(attn, tf.transpose(attn, [0, 2, 1]))
        eye = tf.reshape(tf.tile(tf.eye(self.hps.matrix_num), [self.hps.batch_size, 1]),
                         [-1, self.hps.matrix_num, self.hps.matrix_num])
        self.P = tf.reduce_mean(tf.square(tf.norm(aa_t - eye, axis=[-2, -1], ord='fro')))

    def dec_init_with_mechanism(self):
        # [Batch_size * mechanism_num, mechanism_dim] with [mec1, mec2 .... mecn, mec1, ..... mec n]
        tiled_mechanism = tf.tile(self.matrix, [self.hps.batch_size, 1])

        # [batch_size * mechanism_num, hidden_dim] with [state1, state1, ..... state 2, state2 ... state n, state n]
        tiled_dec_init_state = tf.reshape(tf.tile(self.dec_init_state_small, [1, self.hps.matrix_num]),
                                          [self.hps.batch_size * self.hps.matrix_num, self.hps.hidden_dim])

        # [B * mech_num, dec_hidden_dim] with [state1 + mech1, state1 + mech2 ...]
        self.dec_init_state_cloned = tf.concat([tiled_mechanism, tiled_dec_init_state], axis=1)

    def add_loss_calculation(self):
        dec_tgt_batch = tf.tile(self.dec_tgt_batch, [1, self.hps.matrix_num])
        dec_tgt_batch = tf.reshape(dec_tgt_batch, [self.hps.batch_size * self.hps.matrix_num, self.hps.max_dec_len])

        dec_pad_masks = tf.tile(self.dec_pad_masks, [1, self.hps.matrix_num])
        dec_pad_masks = tf.reshape(dec_pad_masks, [self.hps.batch_size * self.hps.matrix_num, self.hps.max_dec_len])

        seq_loss = tf.contrib.seq2seq.sequence_loss(
            tf.stack(self.dec_vocab_scores, axis=1),
            dec_tgt_batch, weights=dec_pad_masks, average_across_batch=False)

        seq_loss = tf.reshape(seq_loss, [self.hps.batch_size, self.hps.matrix_num])
        min_loss = tf.math.reduce_min(seq_loss, axis=1)
        self.seq_loss = tf.reduce_mean(min_loss)
        self.min_loss_index = tf.argmin(seq_loss, axis=1)

        self.loss = self.seq_loss + self.hps.emb_min_coeff * self.P

        tf.summary.scalar('penalty', self.P)
        tf.summary.scalar('loss_seq', self.seq_loss)
        tf.summary.scalar('loss', self.loss)

    def add_decoder(self):
        cell_fw1 = tf.nn.rnn_cell.GRUCell(self.hps.decoder_hidden_dim, kernel_initializer=self.rand_unif_init)
        cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=self.keep_rate)

        if self.hps.mode == 'decode':
            emb_dec_inp = self.emb_dec_inp
            enc_pad_mask = self.enc_pad_masks
            self.dec_init_state = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.decoder_hidden_dim],
                                                 name='placeholder')

        else:
            self.dec_init_state = self.dec_init_state_cloned  # [B1 + S1, B1 + S2, B1 + S3 ....]

            emb_dec_inp = tf.stack(self.emb_dec_inp)
            tiled_dec_inp = tf.tile(emb_dec_inp, [1, 1, self.hps.matrix_num])
            reshape_dec_inp = tf.reshape(tiled_dec_inp, [self.hps.max_dec_len, self.hps.batch_size *
                                                         self.hps.matrix_num, self.hps.embed_dim])

            emb_dec_inp = tf.unstack(reshape_dec_inp, axis=0)
            enc_pad_mask = tf.reshape(tf.tile(self.enc_pad_masks, [1, self.hps.matrix_num]),
                                      [-1, tf.shape(self.enc_pad_masks)[1]])

            self.encoder_outputs = tf.reshape(tf.tile(self.encoder_outputs, multiples=[1, self.hps.matrix_num, 1]),
                                              [self.hps.batch_size * self.hps.matrix_num, -1,
                                               2 * self.hps.hidden_dim])



        self.dec_outputs, dec_out_states, attn_dists = attention_decoder(emb_dec_inp, self.dec_init_state,
                                                                         self.encoder_outputs, enc_pad_mask,
                                                                         cell_fw1,
                                                                         initial_state_attention=self.hps.mode == 'decode')
        self.dec_out_states = (dec_out_states, [])
        self.attn_dists = (attn_dists, [], [])


    def add_embedding(self):
        self.embedding_matrix = make_custom_embedding_matrix(self.vocab, self.hps)
        self.emb_enc_inp = tf.nn.embedding_lookup(self.embedding_matrix, self.enc_batch)
        self.emb_dec_inp = [tf.nn.embedding_lookup(self.embedding_matrix, x) for x in
                            tf.unstack(self.dec_inp_batch, axis=1)]

    def run_encoder(self, batch, sess, selected_mechanism=-1):
        assert 0 <= selected_mechanism < self.hps.matrix_num
        feeddict = self.make_feeddict(batch, encoder_only=True)
        feeddict[self.keep_rate] = 1.0
        enc_states, dec_init_state = sess.run([self.encoder_outputs, self.dec_init_state_cloned], feeddict)

        dec_init_state = dec_init_state[selected_mechanism]

        return enc_states, dec_init_state

    def add_output_projection(self):
        with tf.variable_scope('dec_output_projection'):
            w = tf.get_variable('w1', [self.hps.decoder_hidden_dim, len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)
            v = tf.get_variable('v1', [len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)

            dec_flatten = tf.reshape(tf.stack(self.dec_outputs), [-1, self.hps.decoder_hidden_dim])

            dec_vocab_scores = tf.nn.xw_plus_b(dec_flatten, w, v)
            if self.hps.mode == 'decode':
                dec_vocab_scores = tf.reshape(dec_vocab_scores,
                                              [-1, self.hps.batch_size, len(self.vocab.words)])
            else:
                dec_vocab_scores = tf.reshape(dec_vocab_scores,
                                              [-1, self.hps.batch_size * self.hps.matrix_num, len(self.vocab.words)])

            dec_vocab_dists = tf.nn.softmax(dec_vocab_scores)

            self.dec_vocab_scores = tf.unstack(dec_vocab_scores)
            self.dec_vocab_dists = tf.unstack(dec_vocab_dists)

    def run_step(self, batch, sess, is_train=False, freeze_layer=False):
        feeddict = self.make_feeddict(batch)
        to_return = {
            'summaries': self.summaries,
            'loss': self.loss,
            'global_step': self.global_step
        }
        if is_train:
            to_return['selected_emb_idx'] = self.min_loss_index
            if freeze_layer:
                feeddict[self.freeze_layer] = freeze_layer
                to_return['train_op'] = self.train_op
                # to_return['train_op'] = self.train_op_freeze
            else:
                to_return['train_op'] = self.train_op

        return sess.run(to_return, feeddict)
