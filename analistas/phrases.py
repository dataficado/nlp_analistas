# coding: utf-8
"""Modulo para transformacion de corpus a vector-espacio phrases."""
import datetime
import logging
import os
import string
import sys
import time

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import WordPunctTokenizer
import nltk.data
import pandas as pd

import helpers as hp


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    hoy = datetime.date.today()
    corrida = "{:%Y-%m-%d}".format(hoy)
    cmd = sys.argv[1]

    dir_curr = os.path.abspath('.')
    dir_input = os.path.join(dir_curr, 'extraction')
    dir_output = os.path.join(dir_curr, 'phrases')
    dir_logs = os.path.join(dir_output, 'logs')
    os.makedirs(dir_logs, exist_ok=True)

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

    if cmd == 'all':
        gcols = ['idioma']
    else:
        gcols = ['corpus', 'idioma']

    grouped = procesados.groupby(gcols)
    for grupo, df in grouped:
        logging.info('{} filas en grupo {}'.format(len(df.index), grupo))

        if cmd == 'all':
            savepath = os.path.join(dir_output, grupo)
        else:
            savepath = os.path.join(dir_output, *grupo)

        paths = df['filepath']

        if 'es' in grupo:
            os.makedirs(savepath, exist_ok=True)

            corpus = hp.get_corpus(paths, sntt, wdt, stops=punct)
            big = Phrases(corpus)
            bigpath = os.path.join(savepath, 'big')
            big.save(bigpath)

            big = Phraser(big)

            corpus = hp.get_corpus(paths, sntt, wdt, stops=punct)
            trig = Phrases(big[corpus])
            trigpath = os.path.join(savepath, 'trig')
            trig.save(trigpath)

            trig = Phraser(trig)

            corpus = hp.get_corpus(paths, sntt, wdt, stops=punct)
            quad = Phrases(trig[big[corpus]])
            quadpath = os.path.join(savepath, 'quad')
            quad.save(quadpath)

    fin = time.time()
    secs = fin - inicio

    logging.info('{m:.2f} minutos'.format(m=secs / 60))


if __name__ == '__main__':
    main()
