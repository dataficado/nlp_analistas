# Sentimiento Relativo de Analistas de Mercado

## Basado en NLP y Listas de palabras predefinidas
### Contenido principal
* custom.txt - archivo de palabras a excluir (custom stopwords)
* helpers.py - Codigo para funciones y variables de uso comun
* extraction.py - Codigo para extraer texto de archivos binarios
* phrases.py - Codigo para crear tokens basado en collocations
* sentiment.py - Codigo para calculo de Indicador de Sentimiento
* topics.py - Codigo para transformacion de corpus a Topicos
* visual_sentiment.py - Codigo para visualizacion de Sentimiento
* visual_topics.py - Codigo para visualizaciones de Topicos

### Orden de ejecucion
Generalmente en el orden listados, exceptuando helpers.py que no se ejecuta directamente
Command Line Arguments (*requerido* **opcional**):
* python extraction.py *path_to_corpus*
* python phrases.py
* python sentiment.py *path_to_wordlist_json* **stems**
* python topics.py **yyyy-mm-dd (inicio)** **yyyy-mm-dd (final)** **mejora | deterioro**
* python visual_sentiment.py **yyyy-mm (inicio)** **yyyy-mm (final)**
* python visual_topics.py **mejora | deterioro**

## Requerimientos
* Ver environment.yml
* Requiere haber descargado NLTK Data (Instrucciones en http://www.nltk.org/data.html)

## Observaciones
### Extraccion
* Usa Tika Server (tika-server-1.13.jar) para extraer texto de documentos. El paquete de python descarga la ultima version si no ve que este corriendo. Yo prefiero abrir tika-server-1.13.jar directamente, para no tener que descargar. Puede descargar la ultima version del server en http://tika.apache.org/download.html

### wordclouds
* Para generar wordclouds se usa https://github.com/amueller/word_cloud
* Fuente de fonts http://www.fontsquirrel.com/fonts/list/language/spanish

### Sentimiento Relativo
* Basado en Relative Difference para el calculo.
* Relative Difference - Often used when two numbers reflect change in single underlying entity.
* Absolute Difference |Δ| scaled by some function f(x,y)
* https://en.wikipedia.org/wiki/Relative_change_and_difference
