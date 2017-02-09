# coding: utf-8
"""Modulo para calculo de Indicador de Sentimiento."""
from collections import Counter
import datetime
import json
import logging
import os
import string
import sys
import time

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import nltk.data
import numpy as np
import pandas as pd

import helpers as hp


def score_sentence(tokens, s1, s2):
    """
    Calcula sentimiento de una frase.

    :param tokens: list
    :param s1: set
    :param s2: set

    :yield: tuple
    """
    fd = Counter(tokens)

    emo1 = sum([c for w, c in fd.items() if any(s in w.split('_') for s in s1)])
    emo2 = sum([c for w, c in fd.items() if any(s in w.split('_') for s in s2)])
    emosum = emo1 + emo2

    emodiff = emo1 - emo2
    emoavg = emosum / 2

    try:
        score = (emodiff / emoavg)
        emosent = 1
    except ZeroDivisionError:
        score = np.nan
        emosent = 0
    except Exception as e:
        score = np.nan
        emosent = 0
        logging.info('ERROR inesperado calculando sentimiento: {}'.format(e))

    return score, emosent


def score_doc(path, wsets, mods, stk, wtk, stp, **kwargs):
    """
    Calcula sentimiento de documento en path.

    :param path: str
    :param wsets: dict
    :param mods: dict
    :param stk: Tokenizer
    :param wtk: WordPunctTokenizer
    :param stp: set

    :yield: tuple
    """
    fields = ['score', 'emosents', 'sents']
    s1 = wsets['mejora']
    s2 = wsets['deterioro']
    results = {}
    r = []

    for sent in hp.transform_sents(path, mods, stk, wtk, stp):
        tokens = hp.tokenize_sent(sent, wtk, **kwargs)
        # tokens = [stmr.stem(wd) for wd in tokens]
        # para reincluir stemmer :param stmr: SnowballStemmer
        # e importar from nltk.stem import SnowballStemmer
        if tokens:
            score, emosent = score_sentence(tokens, s1, s2)
            r.append((score, emosent, 1))

    res = [e for e in zip(*r)]
    for i, f in enumerate(fields):
        if i == 0:
            results[f] = np.nanmean(res[i])
        else:
            results[f] = np.nansum(res[i])

    return results


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    hoy = datetime.date.today()
    corrida = "{:%Y-%m-%d}".format(hoy)
    cmd = sys.argv[1]
    wdlist = sys.argv[2]

    dir_curr = os.path.abspath('.')
    dir_input = os.path.join(dir_curr, 'extraction')
    dir_output = os.path.join(dir_curr, 'sentiment')
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
    # stmmr = SnowballStemmer('spanish')

    punct = set(string.punctuation)
    span = set(stopwords.words('spanish'))

    custom = set(l.strip() for l in open('custom.txt', encoding='utf-8'))
    stops = span.union(custom)

    with open(wdlist, encoding='utf-8') as infile:
        banrep = json.load(infile, encoding='utf-8')

    # stems = {}
    # for cat in banrep.keys():
    #     words = banrep[cat]
    #     stems[cat] = set([stmmr.stem(w) for w in words])

    col_names = ['filepath', 'corpus', 'archivo', 'idioma', 'creacion']
    converter = dict(idioma=lambda x: 'es' if x == 'es' else 'other')
    procesados = pd.read_csv(os.path.join(dir_input, 'procesados.csv'),
                             header=None,
                             names=col_names,
                             encoding='utf-8',
                             converters=converter,
                             parse_dates=['creacion'],
                             infer_datetime_format=True
                             )

    n0 = len(procesados.index)
    logging.info('{} docs en archivo'.format(n0))

    procesados.dropna(subset=['creacion'], inplace=True)
    n1 = len(procesados.index)
    logging.info('{} docs perdidos por no tener fecha'.format(n0 - n1))

    dfs = []

    if cmd == 'all':
        gcols = ['idioma']
    else:
        gcols = ['corpus', 'idioma']

    grouped = procesados.groupby(gcols)
    for grupo, df in grouped:
        logging.info('{} filas en grupo {}'.format(len(df.index), grupo))

        if cmd == 'all':
            loadpath = os.path.join(dir_models, grupo)
        else:
            loadpath = os.path.join(dir_models, *grupo)

        paths = df['filepath']

        if 'es' in grupo:
            mods = {}

            for m in ['big', 'trig', 'quad']:
                modelpath = os.path.join(loadpath, m)
                model = Phrases().load(modelpath)
                ph_model = Phraser(model)
                mods[m] = ph_model

            resu = paths.apply(score_doc,
                               args=(banrep, mods, sntt, wdt, punct),
                               wdlen=3, stops=stops, alphas=False, fltr=5)

            new = resu.apply(pd.Series)
            resultado = pd.concat([df, new], axis=1)

            dfs.append(resultado)

    sentiment = pd.concat(dfs)
    n2 = len(sentiment.index)
    logging.info('Sentimiento calculado para {} docs'.format(n2))
    savepath = os.path.join(dir_output, 'sentiment.csv')
    sentiment.to_csv(savepath, encoding='utf-8', index=False)

    fin = time.time()
    secs = fin - inicio

    logging.info('{m:.2f} minutos'.format(m=secs / 60))


if __name__ == '__main__':
    main()
