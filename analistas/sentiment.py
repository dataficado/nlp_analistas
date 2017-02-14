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
import warnings

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
    result = {}
    r = []

    corpus = os.path.basename(os.path.dirname(path))
    doc = os.path.basename(path)

    for sent in hp.transform_sents(path, mods, stk, wtk, stp):
        tokens = hp.tokenize_sent(sent, wtk, **kwargs)
        # tokens = [stmr.stem(wd) for wd in tokens]
        # para reincluir stemmer :param stmr: SnowballStemmer
        # e importar from nltk.stem import SnowballStemmer
        if tokens:
            score, emosent = score_sentence(tokens, s1, s2)
            r.append((score, emosent, 1))

    res = [e for e in zip(*r)]
    if len(res) == 3:
        for i, f in enumerate(fields):
            if i == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    result[f] = np.nanmean(res[i])
            else:
                result[f] = np.nansum(res[i])
    else:
        for i, f in enumerate(fields):
            result[f] = np.nan

        logstr = '({}, {}) sin score'.format(corpus, doc)
        logging.info(logstr)

    return result


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    ahora = datetime.datetime.now()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(ahora)
    wdlist = sys.argv[1]

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

    names = ['filepath', 'fuente', 'archivo', 'idioma', 'creacion']
    converter = dict(idioma=lambda x: 'es' if x == 'es' else 'other')

    refpath = os.path.join(dir_input, 'metadata.csv')
    if not os.path.isfile(refpath):
        refpath = os.path.join(dir_input, 'procesados.csv')

    procesados = hp.load_reference(refpath, names, converter)

    procesados['creacion'] = pd.to_datetime(procesados['creacion'],
                                            errors='coerce', utc=True,
                                            infer_datetime_format=True)

    n0 = len(procesados.index)
    logging.info('{} docs en archivo'.format(n0))

    procesados.dropna(subset=['creacion'], inplace=True)
    n1 = len(procesados.index)
    logging.info('{} docs perdidos por no tener fecha'.format(n0 - n1))

    dfs = []
    gcols = ['idioma']
    grouped = procesados.groupby(gcols)
    for grupo, df in grouped:
        logging.info('{} docs en grupo {}'.format(len(df.index), grupo))
        paths = df['filepath']
        loadpath = os.path.join(dir_models, grupo)

        if 'es' in grupo:
            mods = {}

            for m in ['big', 'trig', 'quad']:
                modelpath = os.path.join(loadpath, m)
                model = Phrases().load(modelpath)
                ph_model = Phraser(model)
                mods[m] = ph_model

            logstr = 'Calculando sentimiento de docs en {}'.format(grupo)
            logging.info(logstr)

            resu = paths.apply(score_doc,
                               args=(banrep, mods, sntt, wdt, punct),
                               wdlen=3, stops=stops, alphas=True, fltr=5)

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
