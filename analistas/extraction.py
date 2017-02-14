# coding: utf-8
"""Modulo para extraer texto de archivos binarios."""
from pathlib import PurePath
import csv
import datetime
import json
import logging
import os
import sys
import time

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
    info = {}
    parsed = parser.from_file(filepath, server)
    contenido = parsed.get('content')
    metadata = parsed.get('metadata')
    idioma = language.from_buffer(contenido)

    info['contenido'] = contenido
    info['metadata'] = metadata
    info['idioma'] = idioma

    return info


def process_file(filepath, outfile, procfile):
    """
    Hacer parsing de archivo en filepath, guardando resultado en outfile.

    :param filepath: str
    :param outfile: str
    :param procfile: str

    :return: None
    """
    info = parse_with_tika(filepath)

    if info:
        idioma = info.get('idioma')
        metadata = info.get('metadata')
        if metadata:
            creacion = metadata.get('Creation-Date')
        else:
            creacion = ''

        with open(outfile, "w", encoding='utf-8') as out:
            json.dump(info, out, ensure_ascii=False)

        fuente = os.path.basename(os.path.dirname(filepath))
        archivo = os.path.basename(filepath)

        with open(procfile, 'a', newline='', encoding='utf-8') as out:
            writer = csv.writer(out, delimiter=',')
            datos = [outfile, fuente, archivo, idioma, creacion]
            writer.writerow(datos)


def main():
    """Unificar en main para poder ejecutar despues desde otro script."""
    inicio = time.time()
    ahora = datetime.datetime.now()
    corrida = "{:%Y-%m-%d-%H%M%S}".format(ahora)

    dir_curr = os.path.abspath('.')
    dir_input = sys.argv[1]
    dir_input_parts = PurePath(dir_input)
    dir_input_len = len(dir_input_parts.parts)

    dir_output = os.path.join(dir_curr, 'extraction')
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

    bien = 0
    mal = 0

    formatos = ('.pdf', '.txt', '.doc', '.docx', '.ppt', '.pptx')
    procfile = os.path.join(dir_output, 'procesados.csv')

    for dirpath, dirnames, filenames in os.walk(dir_input):
        folders = PurePath(dirpath)
        folders = folders.parts[dir_input_len:]
        savepath = os.path.join(dir_output, *folders)
        os.makedirs(savepath, exist_ok=True)

        for filename in filenames:
            if filename.lower().endswith(formatos):
                filepath = os.path.join(dirpath, filename)
                decofile = unidecode(filename)
                newpath = os.path.join(dirpath, decofile)
                os.replace(filepath, newpath)
                nombre = decofile.rsplit('.', maxsplit=1)[0]
                outfile = os.path.join(savepath, nombre + '.json')
                presente = os.path.isfile(outfile)
                if not presente:
                    try:
                        process_file(newpath, outfile, procfile)
                        bien += 1
                    except Exception as e:
                        mal += 1
                        logging.info('{f}: {e}'.format(f=filename, e=e))

    fin = time.time()
    secs = fin - inicio

    logging.info(
        '{m:.2f} mins, {e} OK y {p} mal'.format(m=secs / 60, e=bien, p=mal))


if __name__ == '__main__':
    main()
