import os
import time
from pprint import pprint
from copy import deepcopy
import json
import numpy as np
import tensorflow as tf
import utils
from data_loader import RvsBatch
from models.basemodel import BaseModel

class BeamsearchDecoder:
    def __init__(self, model, batcher, vocab, ckpt_id=None, fw_sess=None, bw_model=None, bw_sess=None, bidi_ckpt_path=None):
        self.model = model
        self.bw_model = model
        self.batcher = batcher
        self.vocab = vocab
        self.sess = tf.Session(config=utils.gpu_config()) if fw_sess is None else fw_sess
        self.sess2 = bw_sess
        self.bw_model = bw_model

        if bw_model is None:
            ckpt_path = utils.load_ckpt(self.model.hps, self.model.saver, self.sess)
            print('Checkpoint path name: {}'.format(ckpt_path))
            ckpt_name = 'ckpt-' + ckpt_path.split('-')[-1]
        else:
            ckpt_name = 'ckpt-' + bidi_ckpt_path.split('-')[-1]
        self.decode_dir = os.path.join(model.hps.model_path, make_decode_dir_name(ckpt_name, model.hps))

        if os.path.exists(self.decode_dir):
            pass
        else:
            os.makedirs(self.decode_dir)

    def decode(self):
        result_dict = {}
        decoded_cid = []

        counter = 0
        print("START")
        while True:
            time.sleep(0.5)
            batch = self.batcher.next_batch()
            if batch == 'FINISH':
                print("Final batch")
                break
            assert len(list(set(batch.cids))) == 1  # same sample for one batch
            original_text = batch.original_enc_text[0]
            cid, pid, ppid = batch.cids[0], batch.pids[0], batch.ppids[0]
            original_dec_text = batch.original_dec_text[0]
            if cid in decoded_cid: continue

            result_dict[cid] = {'text': original_text, 'pers': {pid: {ppid: original_dec_text}}, 'decoded': {}}
            if self.model.hps.model in ['multiMech']:
                # selected_ids = find_ids_by_prior(sess=self.sess, model=self.model, batch=batch)
                selected_ids = [_ for _ in range(self.model.hps.mechanism_num)]
            elif self.model.hps.model in ['posterior']:
                selected_ids = find_ids_by_prior(sess=self.sess, model=self.model, batch=batch)
                # selected_ids = [_ for _ in range(self.model.hps.matrix_num)]
            elif self.model.hps.model == 'multi_head':
                selected_ids = [_ for _ in range(self.model.hps.multihead_num)]
            elif self.model.hps.model == 'mmpms':
                selected_ids = [_ for _ in range(self.model.hps.mmpms_num)]
            elif self.model.hps.model in ['embavg', 'embmin']:
                selected_ids = [_ for _ in range(self.model.hps.matrix_num)]
            else:
                selected_ids = [-1]
            for idx,ids in enumerate(selected_ids):
                best_hyps = run_beamsearch(self.sess, self.model, self.vocab, batch, bidi_model=self.bw_model, bidi_sess=self.sess2, selected_id=ids)
                if self.model.hps.model == 'vanilla' or self.bw_model is not None:
                    for hyp_idx in range(len(best_hyps)):  #
                        output_ids = [int(t) for t in best_hyps[hyp_idx].tokens]
                        decoded_toks = utils.ids2tokens(output_ids, self.vocab)
                        decoded_text = ' '.join(decoded_toks)
                        decoded_log_avg_prob = round(best_hyps[hyp_idx].avg_log_prob, 5)
                        decoded_text += ' {}'.format(decoded_log_avg_prob)
                        result_dict[cid]['decoded'][hyp_idx] = decoded_text
                else:
                    output_ids = [int(t) for t in best_hyps[0].tokens]
                    decoded_toks = utils.ids2tokens(output_ids, self.vocab)
                    decoded_text = ' '.join(decoded_toks)
                    decoded_log_avg_prob = round(best_hyps[0].avg_log_prob, 5)
                    decoded_text += ' {}'.format(decoded_log_avg_prob)
                    result_dict[cid]['decoded'][int(ids)] = decoded_text
            print("#" * 50)
            print(counter)
            print('TEXT: {}'.format(original_text))
            pprint(result_dict[cid]['decoded'])
            print("#" * 50)
            decoded_cid.append(cid)
            counter += 1
        with open(os.path.join(self.decode_dir, 'summary.json'), 'w') as f:
            json.dump(result_dict, f)




class Hypothesis():
    def __init__(self, tokens, log_probs, state, attn_dists, enc_tokens, bidi_log_probs=[0.0]):
        self.enc_tokens = enc_tokens
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.bidi_log_probs = bidi_log_probs

    def extend(self, token, log_prob, state, attn_dist, bidi_log_probs=0.0):
        return Hypothesis(tokens = self.tokens + [token],
                          log_probs = self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          enc_tokens=self.enc_tokens,
                          bidi_log_probs=self.bidi_log_probs + [bidi_log_probs])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)

    @property
    def rvs_log_prob(self):
        return sum(self.bidi_log_probs)

    @property
    def rvs_avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.rvs_log_prob / len(self.enc_tokens)


def beam_search_hyps(sess, model, vocab, batch, dec_in_state, enc_states, bidi_model=None, selected_ids=-1):
    if model.hps.model == 'mmi_bidi': assert bidi_model is not None
    beg_tok_id = vocab.beg_id
    results = []
    steps = 0

    hyps = [Hypothesis(tokens=[beg_tok_id], log_probs=[0.0], state=dec_in_state[i], attn_dists=[], enc_tokens=batch.original_enc_text[i].split()) for i in
            range(model.hps.beam_size)]

    tlist = []
    while steps < 50:
        latest_tokens = [h.latest_token for h in hyps]

        states = [h.state for h in hyps]
        beg_time = time.time()
        topk_ids, topk_log_probs, new_states, attn_dists = model.decode_onestep(batch=batch,
                                                                                sess=sess,
                                                                                latest_tokens=latest_tokens,
                                                                                enc_states=enc_states,
                                                                                dec_init_states=states,
                                                                                first_step=len(hyps[0].tokens) == 1,
                                                                                selected_ids=selected_ids)
        tlist.append(time.time() - beg_time)

        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        for i in range(num_orig_hyps):
            h, new_state, attn_dist = hyps[i], new_states[i], attn_dists[i]
            for j in range(model.hps.beam_size*2):
                new_hyp = h.extend(token=topk_ids[i,j],
                                   log_prob=topk_log_probs[i,j],
                                   state=new_state,
                                   attn_dist=attn_dist)
                all_hyps.append(new_hyp)

        hyps = []
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.eos_id:
                if steps >= 5:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == model.hps.beam_size:  # or len(results) == model.hps.beam_size:
                break
        steps += 1
    return hyps, results


def reverse_beam_search_hyps(sess, model, batch, vocab, results):
    res = [{'inp': ' '.join(res.enc_tokens), 'tgt': ' '.join(utils.ids2tokens(res.tokens, vocab))} for res in results]
    batch = RvsBatch(res, batch.hps, vocab)
    enc_states, dec_init_state = model.run_encoder(batch, sess)
    dec_init_state = [dec_init_state] * model.hps.beam_size

    prob_dists = model.decode_bidi_seq(batch, sess, enc_states, dec_init_state)

    for batch_idx, batch_prob in enumerate(prob_dists):  # one_step: [max_dec_step, vocab_size]
        ids = vocab.text2ids(res[batch_idx]['inp'].split())
        for step in range(batch.dec_lens[batch_idx] - 1):  # For each batch's step
            corres_ids = ids[step]
            one_prob = batch_prob[step][corres_ids]
            results[batch_idx].bidi_log_probs.append(one_prob)

    return results

def find_ids_by_prior(sess, model, batch):
    assert model.hps.model in ['posterior', 'multiMech']
    enc_states, dec_init_state, prior = model.run_encoder(batch, sess, 0)
    print('Prior: {}\n'.format(prior[0]))
    topk_prior = np.argsort(-prior, axis=1)[0]
    iter_idx_list = topk_prior[:3]
    return iter_idx_list


def run_beamsearch(sess, model, vocab, batch, bidi_model=None, bidi_sess=None, selected_id=-1):
    if model.hps.model in ['posterior', 'multi_head', 'multiMech', 'mmpms']:
        assert selected_id != -1

    if model.hps.model in ['posterior', 'multiMech']:
        enc_states, dec_init_state, prior = model.run_encoder(batch, sess, selected_id)
    elif model.hps.model in ['multi_head', 'mmpms', 'embmin', 'embavg']:
        enc_states, dec_init_state = model.run_encoder(batch, sess, selected_id)
    else:
        enc_states, dec_init_state = model.run_encoder(batch, sess)

    dec_init_state = [dec_init_state] * model.hps.beam_size

    hyps, results = beam_search_hyps(sess, model, vocab, batch, dec_init_state, enc_states, bidi_model=bidi_model, selected_ids=selected_id)

    if len(results) == 0: results = hyps

    results = sort_hyps(results)

    if bidi_model is not None and bidi_sess is not None:
        if len(results) < model.hps.beam_size:
            results += sort_hyps(hyps)[:model.hps.beam_size - len(results)]
        if len(results) > model.hps.beam_size:
            results = results[:model.hps.beam_size]
        assert len(results) == model.hps.beam_size
        results = reverse_beam_search_hyps(bidi_sess, bidi_model, batch, vocab, results)
        results = sort_mmi_hyps(results, model.hps.mmi_lambda, model.hps.mmi_gamma)

    return results[:5]


def sort_hyps(results):
    return sorted(results, key=lambda h: h.avg_log_prob, reverse=True)


def sort_mmi_hyps(results, lambda_val, gamma_val):
    return sorted(results, key=lambda h: h.avg_log_prob - h.rvs_avg_log_prob * lambda_val + len(h.tokens) * gamma_val,
                  reverse=True)


def make_decode_dir_name(ckpt_name, hps):
    dirname = '{}_'.format(ckpt_name) if ckpt_name is not None else ''
    dirname += 'beamsize{}_'.format(hps.beam_size)
    return dirname