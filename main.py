from copy import deepcopy
import argparse
from time import time, sleep
import os
import json
import numpy as np
import tensorflow as tf
from data_loader import Batcher
import utils
from models.basemodel import BaseModel
from models.lm import LMModel
from models.emb_min import DiverEmbMin
from beamsearch import BeamsearchDecoder


parser = argparse.ArgumentParser()

# path for data and models storage
parser.add_argument("--parsed_data_path", type=str, default="data/trainable/split/parsed_perspectrum_data.json")
parser.add_argument("--processed_data_path", type=str, default="data/trainable/split/processed_perspectrum_data.json")
parser.add_argument("--split_data_path", type=str, default="data/trainable/split/{}_processed.json")
parser.add_argument('--wikitext_raw_path', type=str, default='data/wikitext/wikitext-103/wiki.{}.tokens')
parser.add_argument("--data_path", type=str, default="data/trainable/split/train_processed.bin", help="Path to binarized train/valid/test data.")
parser.add_argument("--vocab_path", type=str, default="data/vocab.txt", help="Path to vocabulary.")
parser.add_argument("--embed_path", type=str, default="data/emb/glove.6B.300d.txt", help="Path to word embedding.")
parser.add_argument("--custom_embed_path", type=str, default="data/emb/my_words.txt")
parser.add_argument("--model_path", type=str, default="data/log/{}", help="Path to store the models checkpoints.")
parser.add_argument("--exp_name", type=str, default="scratch", help="Experiment name under model_path.")
parser.add_argument("--parser_path", type=str, default="./stanford-corenlp-full-2018-10-05")
parser.add_argument("--pretrain_ckpt_path", type=str, default='./data/log/lm/scratch/train/')
parser.add_argument('--gpu_nums', type=str, default='1', help='gpu id to use')

# models setups
parser.add_argument("--model", type=str, choices=["vanilla", 'lm', 'embmin', 'mmi_bidi'], default="vanilla", help="Different types of models, choose from vanilla, sep_dec, and shd_dec.")
parser.add_argument("--mode", type=str, choices=["train", "lm_train", "decode", 'eval'], help="Whether to run train, eval, or decode", default="train")
parser.add_argument("--min_cnt", type=int, help="word minimum count", default=1)
parser.add_argument("--use_pretrain", type=str, choices=['True', 'False'], default='True')

parser.add_argument('--beam_size', type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_enc_len", type=int, default=50)
parser.add_argument("--max_dec_len", type=int, default=50)
parser.add_argument("--max_lm_len", type=int, default=150)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--rand_unif_init_size", type=float, default=1e-3)
parser.add_argument("--trunc_norm_init_std", type=float, default=1e-3)
parser.add_argument("--max_grad_norm", type=float, default=3)
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--embed_dim", type=int, default=300)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--decoder_hidden_dim", type=int, default=384)
parser.add_argument("--keep_rate", type=float, default=0.8)
parser.add_argument("--max_epoch", type=int, default=25)

parser.add_argument("--matrix_num", type=int, default=10)
parser.add_argument("--matrix_dim", type=int, default=128)
parser.add_argument('--gumbel_temp', type=float, default=0.67)
parser.add_argument('--use_aux_task', type=str, default='True')
parser.add_argument('--kl_coeff', type=float, default=0.5)
parser.add_argument('--aux_coeff', type=float, default=1.0)
parser.add_argument('--reg_coeff', type=float, default=0.05)

parser.add_argument('--multihead_num', type=int, default=5)

parser.add_argument('--mmi_bsize', type=int, default=100)
parser.add_argument('--mmi_lambda', type=float, default=0.5)
parser.add_argument('--mmi_gamma', type=float, default=1.0)

parser.add_argument('--emb_min_coeff', type=float, default=0.5)

args = parser.parse_args()


def train(model, vocab, pretrain_vardicts=None):
    train_data_loader = Batcher(vocab, model.hps.data_path, args)
    valid_data_loader = Batcher(vocab, model.hps.data_path.replace('train_', 'dev_'), args)
    if model.hps.mode == 'lm_train': valid_data_loader = Batcher(vocab, model.hps.data_path.replace('train_', 'valid_'),
                                                                 args)

    with tf.Session(config=utils.gpu_config()) as sess:
        train_logdir, dev_logdir = os.path.join(args.model_path, 'logdir/train'), os.path.join(args.model_path, 'logdir/dev')
        train_savedir = os.path.join(args.model_path, 'train/')
        print("[*] Train save directory is: {}".format(train_savedir))
        if not os.path.exists(train_logdir): os.makedirs(train_logdir)
        if not os.path.exists(dev_logdir): os.makedirs(dev_logdir)
        if not os.path.exists(train_savedir): os.makedirs(train_savedir)

        summary_writer1 = tf.summary.FileWriter(train_logdir, sess.graph)
        summary_writer2 = tf.summary.FileWriter(dev_logdir, sess.graph)

        """
        Initialize with pretrain variables
        """
        if model.hps.use_pretrain:
            assign_ops, uninitialized_varlist = utils.assign_pretrain_weights(pretrain_vardicts)
            sess.run(assign_ops)
            sess.run(tf.initialize_variables(uninitialized_varlist))
        else:
            sess.run(tf.global_variables_initializer())
        posterior = [0 for _ in range(model.hps.matrix_num)]
        prior = [0 for _ in range(model.hps.matrix_num)]
        step = 0
        while True:  # 6978 sample for one epoch
            beg_time = time()

            batch = train_data_loader.next_batch()
            sample_per_epoch = 857899 if 'lm' in model.hps.mode else 6978

            if model.hps.mode == 'lm_train':
                res = model.run_step(batch, sess, is_train=True)
            else:
                res = model.run_step(batch, sess, is_train=True,
                                     freeze_layer=(model.hps.use_pretrain and step < sample_per_epoch / model.hps.batch_size))
            loss, summaries, step = res['loss'], res['summaries'], res['global_step']
            if model.hps.model == 'posterior':
                gumbel = res['posterior']
                gumbel_prior = res['prior']
                selected = np.argsort(-gumbel)
                selected_poste = [int(el[0]) for el in selected]
                selected_prior = [int(el[0]) for el in np.argsort(-gumbel_prior)]
                posterior = [el1 + el2 for el1, el2 in zip(posterior, selected_poste)]
                prior = [el1 + el2 for el1, el2 in zip(prior, selected_prior)]
                print("prior: {}  posterior: {}".format(prior, posterior))
            elif model.hps.model == 'embmin':
                dist = res['selected_emb_idx']
                for tmp in dist: prior[tmp] += 1
                print(prior)


            end_time = time()
            print("{} epoch, {} step, {}sec, {} loss".format(int(step * model.hps.batch_size / sample_per_epoch), step,
                                                             round(end_time - beg_time, 3), round(loss, 3)))
            summary_writer1.add_summary(summaries, step)

            if step % 5 == 0:
                dev_batch = valid_data_loader.next_batch()
                res = model.run_step(dev_batch, sess, is_train=False)
                loss, summaries, step = res['loss'], res['summaries'], res['global_step']
                assert step % 5 == 0
                print("[VALID] {} loss".format(round(loss, 3)))
                summary_writer2.add_summary(summaries, step)

            if step == 10 or step % 2000 == 0:
                model.saver.save(sess, train_savedir, global_step=step)

            if int(step * model.hps.batch_size / sample_per_epoch) > model.hps.max_epoch:
                model.saver.save(sess, train_savedir, global_step=step)
                print("training end")
                break


def main():
    utils.print_config(args)

    if 'train' not in args.mode:
        args.keep_rate = 1.0
    args.use_pretrain = True if args.use_pretrain == 'True' else False
    args.use_aux_task = True if args.use_aux_task == 'True' else False
    if args.mode == 'lm_train':
        args.model = 'lm'
        args.data_path = "./data/wikitext/wikitext-103/processed_wiki_train.bin"
        args.use_pretrain = False

    args.model_path = os.path.join(args.model_path, args.exp_name).format(args.model)
    print(args.model_path)
    if not os.path.exists(args.model_path):
        if 'train' not in args.mode:
            print(args.model_path)
            raise ValueError
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, 'config.json'), 'w', encoding='utf8') as f:
        json.dump(vars(args), f)

    print("Default models path: {}".format(args.model_path))

    print('code start/ {} mode / {} models'.format(args.mode, args.model))
    utils.assign_specific_gpu(args.gpu_nums)

    vocab = utils.Vocab()

    vardicts = utils.get_pretrain_weights(
        args.pretrain_ckpt_path) if args.use_pretrain and args.mode == 'train' else None

    if args.mode == 'decode':
        if args.model == 'mmi_bidi': args.beam_size = args.mmi_bsize
        args.batch_size = args.beam_size

    modelhps = deepcopy(args)
    if modelhps.mode == 'decode':
        modelhps.max_dec_len = 1

    if args.model == 'vanilla':
        model = BaseModel(vocab, modelhps)
    elif args.model == 'mmi_bidi':
        if args.mode == 'decode':
            bw_graph = tf.Graph()
            with bw_graph.as_default():
                bw_model = BaseModel(vocab, args)

            bw_sess = tf.Session(graph=bw_graph, config=utils.gpu_config())

            with bw_sess.as_default():
                with bw_graph.as_default():
                    bidi_ckpt_path = utils.load_ckpt(bw_model.hps, bw_model.saver, bw_sess)

            fw_graph = tf.Graph()
            with fw_graph.as_default():
                modelhps.model_path = modelhps.model_path.replace('mmi_bidi', 'vanilla')
                modelhps.model = 'vanilla'
                fw_model = BaseModel(vocab, modelhps)
            fw_sess = tf.Session(graph=fw_graph)
            with fw_sess.as_default():
                with fw_graph.as_default():
                    ckpt_path = utils.load_ckpt(fw_model.hps, fw_model.saver, fw_sess)
        else:
            model = BaseModel(vocab, modelhps)

    elif args.model == 'lm':
        model = LMModel(vocab, modelhps)
    elif args.model == 'embmin':
        model = DiverEmbMin(vocab, modelhps)
    else:
        raise ValueError
    print('models load end')

    if args.mode in ['train', 'lm_train']:
        train(model, vocab, vardicts)
    elif args.mode == 'decode':
        import time

        if args.model == 'mmi_bidi':
            batcher = Batcher(vocab, bw_model.hps.data_path.replace('train_', 'test_'), args)
            decoder = BeamsearchDecoder(fw_model, batcher, vocab, fw_sess=fw_sess, bw_model=bw_model, bw_sess=bw_sess, bidi_ckpt_path=bidi_ckpt_path)
        else:
            batcher = Batcher(vocab, model.hps.data_path.replace('train_', 'test_'), args)
            decoder = BeamsearchDecoder(model, batcher, vocab)
        decoder.decode()
    elif args.mode == 'eval':
        pass


if __name__ == '__main__':
    main()
