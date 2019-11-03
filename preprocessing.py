import struct
import json
import argparse
import os
from collections import Counter
from stanfordcorenlp import StanfordCoreNLP as nlp
from tensorflow.core.example import example_pb2

parser = argparse.ArgumentParser()
parser.add_argument("--parsed_data_path", type=str, default="data/trainable/split/parsed_perspectrum_data.json")
parser.add_argument("--processed_data_path", type=str, default="data/trainable/split/processed_perspectrum_data.json")
parser.add_argument("--split_data_path", type=str, default="data/trainable/split/{}_processed.json")
parser.add_argument("--data_path", type=str, default="data/trainable/split/train_processed.bin", help="Path to binarized train/valid/test data.")
parser.add_argument("--vocab_path", type=str, default="data/vocab.txt", help="Path to vocabulary.")
parser.add_argument("--embed_path", type=str, default="data/embed/glove.6B.300d.txt", help="Path to word embedding.")
parser.add_argument('--wikitext_raw_path', type=str, default='data/wikitext/wikitext-103/wiki.{}.tokens')
parser.add_argument('--wikitext_processed_path', type=str, default='data/wikitext/wikitext-103/processed_wiki_{}.txt')
parser.add_argument("--custom_embed_path", type=str, default="data/embed/my_words.txt")
parser.add_argument("--model_path", type=str, default="data/log/", help="Path to store the models checkpoints.")
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--min_cnt", type=int, help="word minimum count", default=1)
parser.add_argument("--parser_path", type=str, default="./stanford-corenlp-full-2018-10-05")
args = parser.parse_args()



data_path = './data/perspectrum/'
split_data_path = './data/trainable/split/'

raw_perspectrum_data = {
    'claim_pers_fname': data_path+'perspectrum_with_answers_v1.0.json',
    'pers_pool_fname': data_path+'perspective_pool_v1.0.json',
    'split_fname': data_path+'dataset_split_v1.0.json'
}


def read_pers_pool(fname):
    with open(fname, 'r', encoding='utf8') as f:
        data = json.load(f)

    final_data = dict()
    for idx, sample in enumerate(data):
        pId, text = sample['pId'], sample['text']
        final_data[pId] = text
    return final_data


def read_claim_pers(fname):
    with open(fname, 'r', encoding='utf8') as f:
        data = json.load(f)

    final_data = dict()
    for idx,sample in enumerate(data):
        cId, text, pers = sample['cId'], sample['text'], sample['perspectives']
        final_data[cId] = {
            'text': text,
            'pers': pers
        }
    return final_data


def match_claim_pers(claims, pers_texts):
    for cId in claims:
        text_pers = {}
        pers = claims[cId]['pers']
        for per in pers:
            one_per_texts = []
            pids = per['pids']
            for pid in pids:
                one_per_texts.append(pers_texts[pid])
            text_pers[len(list(text_pers.keys()))] = one_per_texts
        claims[cId]['pers'] = text_pers
    return claims


def make_merged_dataset():
    """
    Read claim and perspective file & merge into one file.
    """
    if os.path.exists(args.parsed_data_path):
        with open(args.parsed_data_path, 'r', encoding='utf8') as f:
            return json.load(f)

    pers = read_pers_pool(raw_perspectrum_data['pers_pool_fname'])
    claims = read_claim_pers(raw_perspectrum_data['claim_pers_fname'])
    matched = match_claim_pers(claims, pers)
    
    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path)
    with open(args.parsed_data_path, 'w', encoding='utf8') as f:
        json.dump(matched, f)
    return matched


def make_processed_dataset(raw_dataset):
    assert os.path.exists(args.parsed_data_path)
    if os.path.exists(args.processed_data_path):
        with open(args.processed_data_path, 'r', encoding='utf8') as f:
            return json.load(f)
    processor = Preprocessor()

    for cid, item in raw_dataset.items():
        assert 'pers' in item and 'text' in item
        text_tok = processor.preprocess(item['text'])

        processed_pers = {}
        for perId in item['pers']:
            processed_per = []
            for one_per in item['pers'][perId]:
                assert isinstance(one_per, str)
                tokens = processor.preprocess(one_per)
                processed_per.append(tokens)
            processed_per = {k: v for k, v in enumerate(processed_per)}
            processed_pers[len(list(processed_pers.keys()))] = processed_per
        raw_dataset[cid]['text'] = text_tok
        raw_dataset[cid]['pers'] = processed_pers

    with open(args.processed_data_path, 'w', encoding='utf8') as f:
        json.dump(raw_dataset, f)
    return raw_dataset


class Preprocessor:
    def __init__(self):
        self.nlp = nlp(args.parser_path)

    def preprocess(self, text):
        assert isinstance(text, str)
        tokens = self.tokenize(text)
        return tokens

    def tokenize(self, text):
        res = [tok.lower() for tok in self.nlp.word_tokenize(text.replace('-', ' '))]
        return res


def split_dataset(processed_dataset):
    """
    Split the preprocessed dataset into train/valid/test.
    """
    with open(raw_perspectrum_data['split_fname'], 'r', encoding='utf8') as f:
        split_guide = json.load(f)

    train, dev, test = {}, {}, {}

    for cId, item in processed_dataset.items():
        assert cId in split_guide
        setname = split_guide[cId]
        if setname == 'train':
            train[cId] = item
        elif setname == 'dev':
            dev[cId] = item
        elif setname == 'test':
            test[cId] = item
        else:
            raise KeyError

    with open(args.split_data_path.format('train'), 'w', encoding='utf8') as f:
        json.dump(train, f)
    with open(args.split_data_path.format('dev'), 'w', encoding='utf8') as f:
        json.dump(dev, f)
    with open(args.split_data_path.format('test'), 'w', encoding='utf8') as f:
        json.dump(test, f)
    return train, dev, test


def build_vocab():
    assert os.path.exists(args.split_data_path.format('train'))

    if os.path.exists(args.vocab_path):
        with open(args.vocab_path, 'r') as f:
            ls = f.readlines()
            vocab = [line.strip() for line in ls]
            vocab = vocab[:args.vocab_size]
            print(str(len(vocab)) + ' words in vocab.')
            return vocab

    with open(args.split_data_path.format('train'), 'r', encoding='utf8') as f:
        data1 = json.load(f)
    with open(args.split_data_path.format('dev'), 'r', encoding='utf8') as f:
        data2 = json.load(f)
    with open(args.split_data_path.format('test'), 'r', encoding='utf8') as f:
        data3 = json.load(f)
    data = dict(data1, **data2)
    data = dict(data, **data3)

    counter = Counter()

    for cid,item in data.items():
        text, pers = item['text'], item['pers']

        all_tokens = []
        for perid, per in pers.items():
            for pid, pertoks in per.items():
                assert isinstance(pertoks, list) and all([isinstance(tok, str) for tok in pertoks])
                all_tokens.extend(pertoks)

        assert all([isinstance(tok, str) for tok in all_tokens])
        counter.update(text + all_tokens)

    vocab = ['<BEG>', '<PAD>', '<EOS>', '<UNK>']
    vocab += [el[0] for el in counter.most_common() if el[1] >= args.min_cnt]
    vocab = vocab[:args.vocab_size]
    print("Total unique token in train: {}".format(len(vocab)))

    counter = Counter()
    """
    Count the Wikitext tokens first.
    """
    with open(args.wikitext_processed_path.format('train'), 'r', encoding='utf8') as f:
        ls = f.readlines()
    for line in ls:
        tokens = [tok for tok in line.strip().split() if tok != '<unk>']
        counter.update(tokens)
    wiki_words = [el[0] for el in counter.most_common()]
    print("Wiki tokens: {}".format(len(wiki_words)))

    """
    Consider original dataset vocab. 
    """
    while len(vocab) < args.vocab_size and len(wiki_words) != 0:
        wiki_tok = wiki_words.pop(0)
        if wiki_tok not in vocab:
            vocab.append(wiki_tok)
    print('Final vocab size: {}'.format(len(vocab)))

    with open(args.vocab_path, 'w') as f:
        f.write('\n'.join(vocab))
    return vocab


def process_wikitext(setname):
    assert os.path.exists(args.wikitext_raw_path.format(setname))
    if os.path.exists(args.wikitext_processed_path.format(setname)): return
    with open(args.wikitext_raw_path.format(setname), 'r', encoding='utf8') as f:
        ls = f.readlines()
    processed_wiki = []
    for line in ls:
        if len(line.strip()) == 0 or '=' in line: continue
        line = line.strip().replace('@-@', '-').replace('@.@', '.').replace('@,@', ',').lower()
        processed_wiki.append(line)
    with open(args.wikitext_processed_path.format(setname), 'w', encoding='utf8') as f:
        f.write('\n'.join(processed_wiki))


def create_wikitext_bin_file(setname):
    processed_fname = args.wikitext_processed_path.format(setname)
    bin_fname = processed_fname.replace('.txt', '.bin')
    assert os.path.exists(processed_fname)
    if os.path.exists(bin_fname): return

    with open(processed_fname, 'r', encoding='utf8') as f:
        ls = [line.strip() for line in f.readlines()]

    with open(bin_fname, 'wb') as f:
        for line in ls:
            example = example_pb2.Example()
            example.features.feature['text'].bytes_list.value.extend([line.encode()])
            example_str = example.SerializeToString()
            str_len = len(example_str)
            f.write(struct.pack('q', str_len))
            f.write(struct.pack('%ds' % str_len, example_str))


def create_bin_file(setname):
    split_fname = args.split_data_path.format(setname)
    bin_fname = args.split_data_path.format(setname).replace('.json', '.bin')

    if os.path.exists(bin_fname): return
    with open(split_fname, 'r', encoding='utf8') as f:
        split_data = json.load(f)

    enc_dat, dec_dat, cids, pids, ppids = [[] for _ in range(5)]

    for k, v in split_data.items():
        cid, enc_text = k, v['text']

        for pid, per in v['pers'].items():
            for ppid, pptext in per.items():
                assert isinstance(int(pid), int) and isinstance(int(ppid), int) and isinstance(pptext, list) and all([isinstance(tok, str) for tok in pptext])
                enc_dat.append(enc_text)
                dec_dat.append(pptext)
                cids.append(k)
                pids.append(pid)
                ppids.append(ppid)

    assert len(enc_dat) == len(dec_dat) == len(cids) == len(pids) == len(ppids)
    import random
    idx_list = random.sample([_ for _ in range(len(ppids))], len(ppids))
    assert len(idx_list) == len(list(set(idx_list)))
    enc_dat = [enc_dat[idx] for idx in idx_list]
    dec_dat = [dec_dat[idx] for idx in idx_list]
    cids = [cids[idx] for idx in idx_list]
    pids = [pids[idx] for idx in idx_list]
    ppids = [ppids[idx] for idx in idx_list]

    with open(bin_fname, 'wb') as f:
        for idx in range(len(enc_dat)):
            enc_text, dec_text, cid, pid, ppid = ' '.join(enc_dat[idx]), ' '.join(dec_dat[idx]), cids[idx], pids[idx], ppids[idx]
            example = example_pb2.Example()
            example.features.feature['enc'].bytes_list.value.extend([enc_text.encode()])
            example.features.feature['dec'].bytes_list.value.extend([dec_text.encode()])
            example.features.feature['cid'].bytes_list.value.extend([cid.encode()])
            example.features.feature['pid'].bytes_list.value.extend([pid.encode()])
            example.features.feature['ppid'].bytes_list.value.extend([ppid.encode()])
            example_str = example.SerializeToString()
            str_len = len(example_str)
            f.write(struct.pack('q', str_len))
            f.write(struct.pack('%ds' % str_len, example_str))


if __name__ == '__main__':
    raw_dataset = make_merged_dataset()
    processed_dataset = make_processed_dataset(raw_dataset)
    train, dev, test = split_dataset(processed_dataset)
    for setname in ['train', 'valid', 'test']:
        process_wikitext(setname)

    build_vocab()

    for setname in ['train', 'dev', 'test']:
        create_bin_file(setname)
    for setname in ['train', 'valid', 'test']:
        create_wikitext_bin_file(setname)
