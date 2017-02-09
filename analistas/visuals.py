# coding: utf-8
"""Modulo para visualizar sentimiento."""
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("poster")


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    hoy = datetime.date.today()
    corrida = "{:%Y-%m-%d}".format(hoy)

    dir_curr = os.path.abspath('.')
    dir_input = os.path.join(dir_curr, 'sentiment')
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
    sentimiento = pd.read_csv(os.path.join(dir_input, 'sentiment.csv'),
                              usecols=cols,
                              encoding='utf-8',
                              index_col='creacion',
                              parse_dates=True,
                              infer_datetime_format=True)

    n0 = len(sentimiento.index)
    logging.info('{} docs en archivo'.format(n0))

    t0 = sys.argv[1]
    t1 = sys.argv[2]

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


if __name__ == '__main__':
    main()
