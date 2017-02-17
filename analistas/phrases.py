# coding: utf-8
"""Modulo para transformacion de corpus a vector-espacio phrases."""
import datetime
import logging
import os
import string
import time

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import WordPunctTokenizer
import nltk.data

import helpers as hp


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    ahora = datetime.datetime.now()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(ahora)

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

    names = ['filepath', 'fuente', 'archivo', 'idioma', 'creacion']
    converter = dict(idioma=lambda x: 'es' if x == 'es' else 'other')

    refpath = os.path.join(dir_input, 'metadata.csv')
    if not os.path.isfile(refpath):
        refpath = os.path.join(dir_input, 'procesados.csv')

    procesados = hp.load_reference(refpath, names, converter)

    gcols = ['idioma']
    grouped = procesados.groupby(gcols)
    for grupo, df in grouped:
        logging.info('{} docs en grupo {}'.format(len(df.index), grupo))
        paths = df['filepath']
        savepath = os.path.join(dir_output, grupo)

        if 'es' in grupo:
            os.makedirs(savepath, exist_ok=True)

            corpus = hp.get_corpus(paths, sntt, wdt,
                                   wdlen=2, stops=punct, alphas=True, fltr=5)
            big = Phrases(corpus)
            bigpath = os.path.join(savepath, 'big')
            big.save(bigpath)
            big = Phraser(big)

            corpus = hp.get_corpus(paths, sntt, wdt,
                                   wdlen=2, stops=punct, alphas=True, fltr=5)
            trig = Phrases(big[corpus])
            trigpath = os.path.join(savepath, 'trig')
            trig.save(trigpath)
            trig = Phraser(trig)

            corpus = hp.get_corpus(paths, sntt, wdt,
                                   wdlen=2, stops=punct, alphas=True, fltr=5)
            quad = Phrases(trig[big[corpus]])
            quadpath = os.path.join(savepath, 'quad')
            quad.save(quadpath)

    fin = time.time()
    secs = fin - inicio

    logging.info('{m:.2f} minutos'.format(m=secs / 60))


if __name__ == '__main__':
    main()
