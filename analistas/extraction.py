# coding: utf-8
"""Modulo para extraer texto de archivos binarios."""
from pathlib import PurePath
import csv
import json
import logging
import os
import sys
import time
import warnings

from tika import parser
from tika import language
from unidecode import unidecode


def parse_with_tika(filepath, server='http://localhost:9998'):
    """
    De un archivo en filepath, extraer contenido, metadata e idioma.

    :param filepath: str
    :param server: str

    :return: dict ('contenido'(str), 'metadata'(dict), 'idioma'(str))
    """
    parsed = parser.from_file(filepath, server)
    contenido = parsed.get('content')
    metadata = parsed.get('metadata')
    idioma = language.from_buffer(contenido)

    return dict(contenido=contenido, metadata=metadata, idioma=idioma)


def include_as_processed(pfile, data):
    """
    Modifica archivo en pfile para registrar procesamiento de data.

    :param pfile: str
    :param data: str

    :return: None
    """
    with open(pfile, 'a', newline='', encoding='utf-8') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(data)


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()

    dir_curr = os.path.abspath('.')
    dir_input = sys.argv[1]
    dir_input_parts = PurePath(dir_input)
    dir_input_len = len(dir_input_parts.parts)

    dir_output = os.path.join(dir_curr, 'extraction')

    log_format = '%(asctime)s : %(levelname)s : %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=log_format,
                        datefmt=log_datefmt, level=logging.INFO)

    bien = 0
    mal = 0

    formatos = ('.pdf', '.txt', '.doc', '.docx', '.ppt', '.pptx')
    pfile = os.path.join(dir_output, 'procesados.csv')

    for dirpath, dirnames, filenames in os.walk(dir_input):
        folders = PurePath(dirpath)
        folders = folders.parts[dir_input_len:]
        savepath = os.path.join(dir_output, *folders)
        os.makedirs(savepath, exist_ok=True)

        for filename in filenames:
            if filename.lower().endswith(formatos):
                filepath = os.path.join(dirpath, filename)
                nombre = filename.rsplit('.', maxsplit=1)[0]

                if not all(ord(char) < 128 for char in filename):
                    deconame = unidecode(filename)
                    os.replace(os.path.join(dirpath, filename),
                               os.path.join(dirpath, deconame))
                    filepath = os.path.join(dirpath, deconame)
                    nombre = deconame.rsplit('.', maxsplit=1)[0]

                    logstr = '{} cambia a {}'.format(filename, deconame)
                    logging.info(logstr)

                outfile = os.path.join(savepath, nombre + '.json')
                presente = os.path.isfile(outfile)

                if not presente:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore",
                                                  category=RuntimeWarning)
                            info = parse_with_tika(filepath)
                        bien += 1
                    except Exception as e:
                        info = {}
                        mal += 1
                        logstr = 'Nada guardado para {}:{}'.format(filename, e)
                        logging.info(logstr)

                    if info:
                        idioma = info.get('idioma')
                        metadata = info.get('metadata')

                        if metadata:
                            creacion = metadata.get('Creation-Date')
                        else:
                            creacion = ''

                        with open(outfile, "w", encoding='utf-8') as out:
                            json.dump(info, out, ensure_ascii=False)

                        datos = filepath, outfile, idioma, creacion

                        include_as_processed(pfile=pfile, data=datos)

    fin = time.time()
    secs = fin - inicio

    logstr = '{:.2f} mins, {} OK y {} mal'.format(secs / 60, bien, mal)
    logging.info(logstr)


if __name__ == '__main__':
    main()
