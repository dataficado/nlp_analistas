# coding: utf-8
"""Modulo para visualizar sentimiento."""
import datetime
import logging
import os
import sys

from gensim import corpora
from gensim.models.ldamodel import LdaModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim as gvis
import seaborn as sns

sns.set_context("poster")


def topic_data(n, tipo, dirin):
    """
    Prepara data para visualizacion de topicos

    :param n: int (numero de topicos)
    :param tipo: str (label de tipo de sentimiento)
    :param dirin: str (directorio de entrada)

    :return: PreparedData
    """
    dictf = os.path.join(dirin, 'dict-{}.dict'.format(tipo))
    dictionary = corpora.Dictionary.load(dictf)
    strfmt = 'Diccionario con {} tokens cargado'.format(len(dictionary.keys()))
    logging.info(strfmt)

    bowmm = os.path.join(dirin, 'bow-{}.mm'.format(tipo))
    bowcorpus = corpora.MmCorpus(bowmm)

    ldaf = os.path.join(dirin, 'model-{}-{}.lda'.format(n, tipo))
    ldamodel = LdaModel.load(ldaf)

    data = gvis.prepare(ldamodel, bowcorpus, dictionary, mds='tsne')

    return data


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    ahora = datetime.datetime.now()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(ahora)

    dir_curr = os.path.abspath('.')
    dir_sent = os.path.join(dir_curr, 'sentiment')
    dir_topi = os.path.join(dir_curr, 'topics', 'es')
    dir_output = os.path.join(dir_curr, 'visuals')
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

    cols = ['filepath', 'creacion', 'emosents', 'score', 'sents']
    sentimiento = pd.read_csv(os.path.join(dir_sent, 'sentiment.csv'),
                              usecols=cols,
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

    aggs = dict(filepath=np.count_nonzero,
                emosents=np.sum, sents=np.sum, score=np.mean)

    df = sentimiento.resample('M').agg(aggs)

    df.plot(y='filepath', figsize=(14, 8), legend=False,
            title="Cantidad de documentos procesados - mensual")
    plt.savefig(os.path.join(dir_output, 'docs.png'))

    df.plot(y='score', figsize=(14, 8), legend=False,
            title="Indicador de Sentimiento de Analistas de Mercado (ISAM)")
    plt.savefig(os.path.join(dir_output, 'sentimiento.png'))

    mensual = os.path.join(dir_output, 'mensual.csv')
    df.to_csv(mensual, encoding='utf-8')

    # visualizacion de topicos
    for n in (10, 25):
        data = topic_data(n, tipo, dir_topi)
        strfmt = 'topics-{}-{}.html'.format(n, tipo)
        html = os.path.join(dir_output, strfmt)
        pyLDAvis.save_html(data, html)


if __name__ == '__main__':
    main()
