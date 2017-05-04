# coding: utf-8
"""Modulo para transformacion de corpus a Topicos."""
import datetime
import logging
import os
import string
import sys
import time

from gensim import corpora
from gensim.models import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phraser
from nltk import FreqDist
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import nltk.data
import numpy as np
import pandas as pd

import helpers as hp


def doc_ngrams(words, n=2):
    """
    Transforma BOW de documento en n-gramas.

    :param words: list of str (palabras de un documento)
    :param n: int

    :return: list of str (n-gramas de un documento)
    """
    tokens = []
    coll = FreqDist(item for item in ngrams(words, n))
    fd = {' '.join(w): c for w, c in coll.items()}

    for word, count in fd.items():
        tokens.extend([word] * count)

    return tokens


def create_models(iter_of_bows, outdir, mask, ng):
    """
    Crea y almacena modelos Dictionary, BOWcorpus, LDA

    :param iter_of_bows: list or pd.Series
    :param outdir: str (directorio de salida)
    :param mask: str (label de tipo de sentimiento)
    :param ng: str (uni, bi, tri, phrase)

    :return: None
    """
    diction = corpora.Dictionary(toks for toks in iter_of_bows)
    unids = [tokid for tokid, freq in diction.dfs.items() if freq == 1]
    diction.filter_tokens(unids)  # remove words that appear only once
    diction.compactify()
    strdict = 'dict-{}-{}gram.dict'.format(mask, ng)
    dictpath = os.path.join(outdir, strdict)
    diction.save(dictpath)

    bow = [diction.doc2bow(toks) for toks in iter_of_bows]
    strbow = 'bow-{}-{}gram.mm'.format(mask, ng)
    bowmm = os.path.join(outdir, strbow)
    corpora.MmCorpus.serialize(bowmm, bow)

    #  LDA transformations
    #  10 passes, no online updates, and i topics
    params = dict(id2word=diction, update_every=0, passes=10)
    for i in (5, 10, 20, 40):
        lda = LdaModel(bow, num_topics=i, **params)
        strlda = 'model-{}-{}-{}gram.lda'.format(i, mask, ng)
        ldapath = os.path.join(outdir, strlda)
        lda.save(ldapath)


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(datetime.datetime.now())

    dir_curr = os.path.abspath('.')
    dir_input = os.path.join(dir_curr, 'sentiment')
    dir_output = os.path.join(dir_curr, 'topics')
    dir_logs = os.path.join(dir_output, 'logs')
    os.makedirs(dir_logs, exist_ok=True)

    dir_models = os.path.join(dir_curr, 'phrases')

    logfile = os.path.join(dir_logs, '{}.log'.format(corrida))
    log_format = '%(asctime)s : %(levelname)s : %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=log_format, datefmt=log_datefmt,
                        level=logging.INFO, filename=logfile, filemode='w')

    punkt = os.path.join('tokenizers', 'punkt', 'spanish.pickle')
    sntt = nltk.data.load(punkt)
    wdt = WordPunctTokenizer()

    punct = set(string.punctuation)
    custom = set(l.strip() for l in open('custom.txt', encoding='utf-8'))
    span = set(stopwords.words('spanish'))
    stops = span.union(custom).union(punct)

    sentimiento = pd.read_csv(os.path.join(dir_input, 'sentiment.csv'),
                              encoding='utf-8', index_col='creacion',
                              parse_dates=True, infer_datetime_format=True)

    n0 = len(sentimiento.index)
    logging.info('{} docs en archivo'.format(n0))

    cmds = sys.argv
    nc = len(cmds)
    if nc == 1:
        t0, t1, tipo = None, None, 'todos'
    elif nc == 2:
        t0, t1, tipo = cmds[1], None, 'todos'
    elif nc == 3:
        t0, t1, tipo = cmds[1], cmds[2], 'todos'
    else:
        t0, t1, tipo = cmds[1], cmds[2], cmds[3]

    sentimiento = sentimiento[t0:t1]
    n1 = len(sentimiento.index)
    strlog = '{} docs dentro de periodo y {} fuera'.format(n1, (n0 - n1))
    logging.info(strlog)

    if nc == 4:
        sentimiento['tipo'] = np.where(sentimiento['score'] > 0, 'mejora',
                                       np.where(sentimiento['score'] < 0,
                                                'deterioro', 'neutral'))

        masktipo = sentimiento['tipo'] == tipo
        sentimiento = sentimiento[masktipo]

        n2 = len(sentimiento.index)
        out = n1 - n2
        strlog = 'Se mantienen {} tipo {} y descartan {}'.format(n2, tipo, out)
        logging.info(strlog)

    grouped = sentimiento.groupby(['idioma'])
    for grupo, df in grouped:
        logging.info('{} docs en grupo {}'.format(len(df.index), grupo))
        paths = df['filepath']
        loadpath = os.path.join(dir_models, grupo)
        savepath = os.path.join(dir_output, grupo)

        if 'es' in grupo:
            os.makedirs(savepath, exist_ok=True)
            mods = {}

            for m in ['big', 'trig']:
                modelpath = os.path.join(loadpath, m)
                model = Phrases().load(modelpath)
                ph_model = Phraser(model)
                mods[m] = ph_model

            docwords = paths.apply(hp.phrased_words,
                                   args=(mods, sntt, wdt, 0.1),
                                   wdlen=3, stops=stops, alphas=True, fltr=5)

            uniwords = paths.apply(hp.simple_words,
                                   args=(sntt, wdt, 0.1),
                                   wdlen=3, stops=stops, alphas=True, fltr=5)

            biwords = uniwords.apply(doc_ngrams, n=2)
            triwords = uniwords.apply(doc_ngrams, n=3)

            # modelos usando frased-words, unigramas, bigramas, trigramas
            create_models(docwords, savepath, tipo, 'phrase')
            create_models(uniwords, savepath, tipo, 'uni')
            create_models(biwords, savepath, tipo, 'bi')
            create_models(triwords, savepath, tipo, 'tri')

    fin = time.time()
    secs = fin - inicio

    logging.info('{m:.2f} minutos'.format(m=secs / 60))


if __name__ == '__main__':
    main()
