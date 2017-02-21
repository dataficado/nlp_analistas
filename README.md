# Sentimiento Relativo de Analistas de Mercado

## Basado en NLP y Listas de palabras predefinidas
Contiene principalmente:

* extraction.py - Codigo para extraer texto de archivos binarios
* helpers.py - Codigo para funciones y variables de uso comun
* phrases.py - Codigo para crear tokens basado en collocations
* sentiment.py - Codigo para calculo de Indicador de Sentimiento
* topics.py - Codigo para transformacion de corpus a Topicos
* visuals.py - Codigo para visualizaciones

## Requerimientos
Ver environment.yml

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

### Doc NLP
* Requiere haber descargado NLTK Data (Instrucciones en http://www.nltk.org/data.html)
