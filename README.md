# NLProlog

This is an implementation of [NLProlog](todo), a method for approaching Question Answering tasks with Prolog-like reasoning over natural language statements. 

At the core of NLProlog is the Prolog interpreter [sPyrolog](https://github.com/leonweber/nlprolog/spyrolog), which can be found in a separate repository.
sPyrolog is a fork of the [Prolog interpreter Pyrolog](https://bitbucket.org/cfbolz/pyrolog/).

![Proof examples](example_proofs.png?raw=true "Title")

## Disclaimer

This is highly experimental research code which is not suitable for production usage. We do not provide warranty of any kind. Use at your own risk.

## Installation

1. The python version has to be at least `3.6`.
2. A working version of sPyrolog has to reside in the project root, if the precompiled version from this repository does not work for you, try recompiling it from the [sPyrolog repository](todo).
3. Install [sent2vec](https://github.com/epfml/sent2vec) and place its [wiki-unigrams model](https://drive.google.com/open?id=0B6VhzidiLvjSa19uYWlLUEkzX3c) in the project root. For reproducing the results on MedHop, the [BioSent2Vec model](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin) has to reside in the project root.
4. Install the `requirements.txt`, e.g. with `pip install -r requirements.txt`.
5. Install pyTorch 1.0


## Usage

Training a model is straight forward.
For instance, to reproduce the results for the `country` predicate of [WikiHop](https://qangaroo.cs.ucl.ac.uk/) use

```bash
python train.py configs/country_sent2vec.json
```

To evaluate the trained model use

```bash
python evaluate.py model/country_sent2vec
```

This will generate a new results file in `results/` which can be analyzed like shown in [this example notebook](notebooks/example_analysis.ipynb).


To train a model on your own data, convert it to the WikiHop format, place it into `data/` and put a configuration file into `configs/`.
