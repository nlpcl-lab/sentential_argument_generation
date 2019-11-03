# sentential_argument_generation

This is an official code for a paper "ArgDiver: Generating Sentential Arguments from Diverse Perspectives on Controversial Topics", accepted in NLP4IF 2019.



## Getting Started

#### Requirements

- tensorflow-gpu==1.8
- numpy==1.16.2
- stanfordcorenlp-3.9.1.1



#### Prerequistes

- Prepare the PERSPECTRUM dataset proposed in [this](<https://www.aclweb.org/anthology/N19-1053>) paper.

- (Optional) Download the Wikitext-103 from [here](<https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/>), proposed in [this]() paper.
   - If you don't pre-train a model, just skip this part and use `--use_pretrain=False` flag when train a model.
   
- Download the Glove word embedding from [here.](<https://nlp.stanford.edu/projects/glove/>)

- After download the above resources, locate them in appropriate directory with following the below `Package Structure`.

#### Commands


Preprocessing
```
python preprocessing.py
```
(Optional) Pretrain the weights of encoder and decoder

```
python main.py --mode=lm_train --model=lm
```
Train model. If you don't pretrain the model in above, please set the --use_pretrain=False
```
python main.py --mode=train --model=[vanilla,embmin]
```
(Optional) To use MMI-bidi for decoding, train both the standard-seq2seq(vanilla) and the reverse-seq2seq model using below command.

```
python main.py --mode=train --model=mmi_bidi
```
Decode using trained model.
```
python main.py --mode=decode --model=[vanilla,mmi_bidi,embmin] --beam_size=10
```
#### Package Structure

```
├── argument-reasoning-comprehension
│     └── beamsearch.py
│     └── data_loader.py
│     └── preprocessing.py
│     └── utils.py
│     └── main.py
│     └── models/
│          └──── _init_.py
│          └──── attention.py
│          └──── basemodel.py
│          └──── emb_min.py
│          └──── lm.py
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

#### Results
```
Input
- We should fear the power of government over the internet.

Outputs:
- National sovereignty would result in a government’s freedom of expression.
- The government should not be celebrated.
- It is a necessary for national security.
- It’s conceivable to the wrong hands.
- The government is a best way to have a universal right to have a universal right to practice.
```


## References

* The implementation of attention mechanism and neural network is based on [this](<https://github.com/XinyuHua/neural-argument-generation>) and [this](<https://github.com/abisee/pointer-generator>) repositories.
* Chen, Sihao, et al. "Seeing Things from a Different Angle: Discovering Diverse Perspectives about Claims." *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. 2019. [paper](<https://www.aclweb.org/anthology/N19-1053>)
