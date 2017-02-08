# coding: utf-8
"""Modulo para variables y funciones de uso comun."""
import json


def extract_text(filepath):
    """
    Extrae texto de archivo en filepath.

    :param filepath: str

    :return: str
    """
    with open(filepath, encoding='utf-8') as infile:
        try:
            info = json.load(infile, encoding='utf-8')
        except Exception as e:
            print('Error en {f}: {e}'.format(f=filepath, e=e))
            info = {}

    text = info.get('contenido')

    return text


def get_sentences(text, sntt):
    """
    Tokeniza un documento en frases.

    :param text: str
    :param sntt: Tokenizer

    :yield: str
    """
    sentences = sntt.tokenize(text)
    yield from sentences


def tokenize_sent(sentence, wdt, wdlen=0, stops=None, alphas=False, fltr=0):
    """
    Tokeniza una frase en palabras.

    :param sentence: str
    :param wdt: WordPunctTokenizer
    :param wdlen: int
    :param stops: set
    :param alphas: Boolean
    :param fltr: int

    :return: list of str
    """
    words = [w.lower() for w in wdt.tokenize(sentence)]
    words = [w for w in words if len(w) > int(wdlen)]
    if stops:
        stops = [w.lower() for w in stops]
        words = [w for w in words if w not in stops]
    if alphas:
        words = [w for w in words if w.isalpha()]

    if not (len(words) > int(fltr)):
        words = []

    return words


def get_doc(path, sntt, wdt, wdlen=0, stops=None, alphas=False, fltr=0):
    """
    Saca cada frase de un documento.

    :param path: str
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param wdlen: int
    :param stops: set
    :param alphas: Boolean
    :param fltr: int

    :yield: list of str (palabras de una frase)
    """
    text = extract_text(path)
    for sentence in get_sentences(text, sntt):
        yield tokenize_sent(sentence, wdt, wdlen, stops, alphas, fltr)


def get_corpus(paths, sntt, wdt, wdlen=0, stops=None, alphas=False, fltr=0):
    """
    Saca cada frase de cada documento en filepaths.

    :param paths: Series
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param wdlen: int
    :param stops: set
    :param alphas: Boolean
    :param fltr: int

    :yield: list of str (palabras de una frase)
    """
    for path in paths:
        yield from get_doc(path, sntt, wdt, wdlen, stops, alphas, fltr)
