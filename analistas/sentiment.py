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

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
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

    emo1 = sum([c for w, c in fd.items() if w in s1])
    emo2 = sum([c for w, c in fd.items() if w in s2])

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

    assert ((-2 <= score <= 2) or score is np.nan)

    return score, emosent


def score_doc(path, ws, sntt, wdt, stmr=None, trim=None, **kwargs):
    """
    Calcula sentimiento de documento en path.

    :param path: str
    :param ws: dict
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param stmr: SnowballStemmer or None
    :param trim: float
    **kwargs: (wdlen, stops, alphas, fltr)

    :yield: tuple
    """
    fields = ['score', 'emosents', 'sents']
    s1 = ws['mejora']
    s2 = ws['deterioro']

    result = {}
    r = []

    for tokens in hp.get_tokenized_sents(path, sntt, wdt, trim, **kwargs):
        if tokens:
            if stmr:
                tokens = [stmr.stem(w) for w in tokens]

            score, emosent = score_sentence(tokens, s1, s2)
            r.append((score, emosent, 1))

    res = [e for e in zip(*r)]
    if len(res) == 3:
        for i, f in enumerate(fields):
            if i == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    result[f] = np.nanmean(res[i])

                assert (result[f] <= 2 or type(result[f]) == np.float64)

            else:
                result[f] = np.nansum(res[i])

    else:

        for i, f in enumerate(fields):
            result[f] = np.nan

        logstr = 'Sin score: {}'.format(path)
        logging.info(logstr)

    return result


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(datetime.datetime.now())
    wdlist = sys.argv[1]

    dir_curr = os.path.abspath('.')
    dir_input = os.path.join(dir_curr, 'extraction')
    dir_output = os.path.join(dir_curr, 'sentiment')
    dir_logs = os.path.join(dir_output, 'logs')
    os.makedirs(dir_logs, exist_ok=True)

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

    stmr = None

    with open(wdlist, encoding='utf-8') as infile:
        banrep = json.load(infile, encoding='utf-8')

    if len(sys.argv) == 3:
        logging.info('Usando stems de listas de palabras...')
        stmr = SnowballStemmer('spanish')
        stems = {}
        for cat in banrep.keys():
            words = banrep[cat]
            stems[cat] = set([stmr.stem(w) for w in words])

        banrep = stems.copy()

    names = ['origen', 'filepath', 'idioma', 'creacion']
    converter = dict(idioma=lambda x: 'es' if x == 'es' else 'other')

    refpath = os.path.join(dir_input, 'metadata.csv')
    if not os.path.isfile(refpath):
        refpath = os.path.join(dir_input, 'procesados.csv')

    procesados = pd.read_csv(refpath, header=None, names=names,
                             converters=converter, encoding='utf-8')

    procesados['creacion'] = pd.to_datetime(procesados['creacion'],
                                            errors='coerce', utc=True,
                                            infer_datetime_format=True)

    n0 = len(procesados.index)
    logging.info('{} docs en archivo'.format(n0))

    procesados.dropna(subset=['creacion'], inplace=True)
    n1 = len(procesados.index)
    logging.info('{} docs perdidos por no tener fecha'.format(n0 - n1))

    dfs = []
    grouped = procesados.groupby(['idioma'])
    for grupo, df in grouped:
        logging.info('{} docs en grupo {}'.format(len(df.index), grupo))
        paths = df['filepath']

        if 'es' in grupo:
            logstr = 'Calculando sentimiento de docs en {}'.format(grupo)
            logging.info(logstr)

            resu = paths.apply(score_doc, args=(banrep, sntt, wdt, stmr, 0.1),
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
