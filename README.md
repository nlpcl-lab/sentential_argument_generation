# sentential_argument_generation

This is a implementation for sentential argument generation.



## Getting Started

#### Requirements

- tensorflow-gpu==1.8
- numpy==1.16.2
- stanfordcorenlp-3.9.1.1



#### Prerequistes

1. Prepare the PERSPECTRUM dataset proposed in [this](<https://www.aclweb.org/anthology/N19-1053>) paper.
2. (Optional) Download the Wikitext-103 from [here](<https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/>), proposed in [this]() paper.
   - If you don't pre-train a model, just skip this part and use `--use_pretrain=False` flag when train a model.
3. Download the Glove word embedding from [here.](<https://nlp.stanford.edu/projects/glove/>)



### commands

```
# Preprocessing
python preprocessing.py
# (Optional) Pretrain the weights of encoder and decoder
python main.py --mode=lm_train
# Train model. If you don't pretrain the model in above, please set the --use_pretrain=False
python main.py --mode=train --model=[vanilla,embmin]
# (Optional) To use MMI-bidi for decoding, train both the standard-seq2seq(vanilla) and the reverse-seq2seq model using below command.
python main.py --mode=train --model=mmi_bidi
# Decode using trained model.
python main.py --mode=decode --model=[vanilla,mmi_bidi,embmin] --beam_size=10
```



#### Package Structure

```
├── argument-reasoning-comprehension
│     └── script.py
│     └── data_helper.py
│     └── preprocessing.py
│     └── util.py
│     └── esim_model.py
│     └── model.py
│     └── data/
│          └──── emb/
│                 └──── glove.6B.300d.txt
│          └──── log/
│          └──── perspectrum/
│                      └──── (Locate the json files of PERSPECTRUM data to here!)
│          └──── wikitext/wikitext-103/
│                      └──── (Locate the Wikitext-103 .tokens files to here!)
│          └──── trainable/
│                      └──── split/	
```



## References

* The implementation of attention mechanism and neural network is based on [this](<https://github.com/XinyuHua/neural-argument-generation>) and [this](<https://github.com/abisee/pointer-generator>) repositories.
* Chen, Sihao, et al. "Seeing Things from a Different Angle: Discovering Diverse Perspectives about Claims." *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. 2019. [paper](<https://www.aclweb.org/anthology/N19-1053>)

## Contributor

[ChaeHun](http://nlp.kaist.ac.kr/~ddehun)

