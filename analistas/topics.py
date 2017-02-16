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
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import nltk.data
import numpy as np
import pandas as pd

import helpers as hp


def get_transformed_docwords(path, mo, sntt, wdt, stp, **kwargs):
    """
    Transforma documento en path en BOW.

    :param path: str
    :param mo: dict
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param stp: set

    :yield: list of str (palabras de un documento)
    """
    tokens = []

    # params de transf_sents deben ser iguales a lo usado en phrases
    for sent in hp.transf_sents(mo, path, sntt, wdt,
                                wdlen=0, stops=stp, alphas=True, fltr=5):

        words = hp.tokenize_sent(sent, wdt, **kwargs)
        tokens.extend(words)

    return tokens


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    ahora = datetime.datetime.now()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(ahora)

    dir_curr = os.path.abspath('.')
    dir_input = os.path.join(dir_curr, 'sentiment')
    dir_output = os.path.join(dir_curr, 'topics')
    dir_logs = os.path.join(dir_output, 'logs')
    os.makedirs(dir_logs, exist_ok=True)

    dir_models = os.path.join(dir_curr, 'phrases')

    logfile = os.path.join(dir_logs, '{}.log'.format(corrida))
    log_format = '%(asctime)s : %(levelname)s : %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=log_format,
                        datefmt=log_datefmt,
                        level=logging.INFO,
                        filename=logfile,
                        filemode='w')

    punkt = os.path.join('tokenizers', 'punkt', 'spanish.pickle')
    sntt = nltk.data.load(punkt)
    wdt = WordPunctTokenizer()

    punct = set(string.punctuation)
    span = set(stopwords.words('spanish'))

    custom = set(l.strip() for l in open('custom.txt', encoding='utf-8'))
    stops = span.union(custom)

    sentimiento = pd.read_csv(os.path.join(dir_input, 'sentiment.csv'),
                              encoding='utf-8',
                              index_col='creacion',
                              parse_dates=True,
                              infer_datetime_format=True)

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
    logging.info('{} docs fuera de periodo'.format(n0 - n1))
    logging.info('{} docs dentro de periodo'.format(n1))

    if nc == 4:
        sentimiento['tipo'] = np.where(sentimiento['score'] > 0, 'mejora',
                                       np.where(sentimiento['score'] < 0,
                                                'deterioro', 'neutral'))

        masktipo = sentimiento['tipo'] == tipo
        sentimiento = sentimiento[masktipo]

        n2 = len(sentimiento.index)
        logging.info('Se descartan {} docs no tipo {}'.format((n1 - n2), tipo))
        logging.info('Se mantienen {} docs tipo {}'.format(n2, tipo))

    gcols = ['idioma']
    grouped = sentimiento.groupby(gcols)
    for grupo, df in grouped:
        logging.info('{} docs en grupo {}'.format(len(df.index), grupo))
        paths = df['filepath']
        loadpath = os.path.join(dir_models, grupo)
        savepath = os.path.join(dir_output, grupo)

        if 'es' in grupo:
            os.makedirs(savepath, exist_ok=True)
            mods = {}

            for m in ['big', 'trig', 'quad']:
                modelpath = os.path.join(loadpath, m)
                model = Phrases().load(modelpath)
                ph_model = Phraser(model)
                mods[m] = ph_model

            docwords = paths.apply(get_transformed_docwords,
                                   args=(mods, sntt, wdt, punct),
                                   wdlen=3, stops=stops, alphas=True, fltr=5)

            diction = corpora.Dictionary(toks for toks in docwords)
            unids = [tokid for tokid, freq in diction.dfs.items() if freq == 1]
            diction.filter_tokens(unids)  # remove words that appear only once
            diction.compactify()
            dictpath = os.path.join(savepath, 'dict-{}.dict'.format(tipo))
            diction.save(dictpath)

            bow = [diction.doc2bow(toks) for toks in docwords]
            bowmm = os.path.join(savepath, 'bow-{}.mm'.format(tipo))
            corpora.MmCorpus.serialize(bowmm, bow)

            #  LDA transformations
            #  10 passes, no online updates, and n topics
            params = dict(id2word=diction, update_every=0, passes=10)
            for n in (10, 25):
                lda = LdaModel(bow, num_topics=n, **params)
                ldapath = os.path.join(savepath,
                                       'model-{}-{}.lda'.format(n, tipo))
                lda.save(ldapath)

    fin = time.time()
    secs = fin - inicio

    logging.info('{m:.2f} minutos'.format(m=secs / 60))


if __name__ == '__main__':
    main()
