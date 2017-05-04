# coding: utf-8
"""Modulo para variables y funciones de uso comun."""
import json
import logging
import re


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
            logging.info('Error cargando {}: {}'.format(filepath, e))
            info = {}

    return info.get('contenido')


def preprocess_text(text, trim=None):
    """
    Preprocesamiento de text.

    :param text: str
    :param trim: float

    :yield: str
    """
    text = re.sub(r'-\n+', '', text)  # para hyphenation final de linea
    text = ' '.join(text.split())  # eliminar multiples espacios

    # Quitar porcentaje de palabras de inicio y final
    if trim:
        toks = text.split()
        size = len(toks)
        i0 = int(size * trim)
        i1 = int(size * (1 - trim))
        text = ' '.join(toks[i0:i1])

    return text


def get_sentences(text, sntt):
    """
    Tokeniza un documento en frases.

    :param text: str
    :param sntt: Tokenizer

    :yield: str
    """
    for sentence in sntt.tokenize(text):
        yield sentence


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
        words = [w for w in words
                 if w.isalpha() or
                 ('_' in w and not any(c.isdigit() for c in w))]

    if not (len(words) > int(fltr)):
        words = []

    return words


def get_tokenized_sents(path, sntt, wdt, trim=None, **kwargs):
    """
    Saca cada frase de un documento.

    :param path: str
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param trim: float
    **kwargs: (wdlen, stops, alphas, fltr)

    :yield: list of str (palabras de una frase)
    """
    text = extract_text(path)
    text = preprocess_text(text, trim)
    for sentence in get_sentences(text, sntt):
        yield tokenize_sent(sentence, wdt, **kwargs)


def get_corpus(paths, sntt, wdt, trim=None, **kwargs):
    """
    Saca cada frase de cada documento en filepaths.

    :param paths: Series
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param trim: float
    **kwargs: (wdlen, stops, alphas, fltr)

    :yield: list of str (palabras de una frase)
    """
    for path in paths:
        for words in get_tokenized_sents(path, sntt, wdt, trim, **kwargs):
            yield words


def get_phrased_sents(mo, path, sntt, wdt, trim=None, **kwargs):
    """
    Transforma cada frase de un documento usando modelos de collocation.

    :param mo: dict
    :param path: str
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param trim: float
    **kwargs: (wdlen, stops, alphas, fltr)

    :yield: str (frase)
    """
    big = mo['big']
    trig = mo['trig']
    sents = get_tokenized_sents(path, sntt, wdt, trim, **kwargs)
    transformed = trig[big[sents]]
    for sent in transformed:
        yield ' '.join(sent)


def phrased_words(path, mo, sntt, wdt, trim=None, **kwargs):
    """
    Transforma documento en path en phrased BOW.

    :param path: str
    :param mo: dict
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param trim: float
    **kwargs: (wdlen, stops, alphas, fltr)

    :return: list of str (palabras de un documento)
    """
    tokens = []

    # params de get_phrased_sents deben ser iguales a lo usado en phrases
    # incluir si se quiere algo diferente a los kwargs de tokenize_sent
    for sent in get_phrased_sents(mo, path, sntt, wdt, trim, **kwargs):
        words = tokenize_sent(sent, wdt, **kwargs)
        tokens.extend(words)

    return tokens


def simple_words(path, sntt, wdt, trim=None, **kwargs):
    """
    Transforma documento en path en BOW.

    :param path: str
    :param sntt: Tokenizer
    :param wdt: WordPunctTokenizer
    :param trim: float
    **kwargs: (wdlen, stops, alphas, fltr)

    :return: list of str (palabras de un documento)
    """
    tokens = []

    for words in get_tokenized_sents(path, sntt, wdt, trim, **kwargs):
        tokens.extend(words)

    return tokens
