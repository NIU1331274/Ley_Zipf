# Formulación probabilística de la Ley de Zipf y estudio estadístico

Código del TFG. Réplica y extensión multilingüe de Moreno-Sánchez, Font-Clos & Corral
(2016): ajuste de las distribuciones de Zipf (f1, f2, f3) a las frecuencias de palabras
de los textos del Standardized Project Gutenberg Corpus (SPGC), y análisis de la
preferencia f1/f2 frente a la riqueza morfológica (fracción de *hapax*) en 11 idiomas.

## Requisitos

- Python ≥ 3.8
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Datos

El análisis parte de los ficheros de conteos del SPGC (`PG*_counts.txt`) y de sus
metadatos. Versión usada: **SPGC-2018-07-18** (Gerlach & Font-Clos, DOI
[10.5281/zenodo.2422561](https://doi.org/10.5281/zenodo.2422561)).

Descarga los datos desde ese enlace y, si la ruta no coincide con la de por defecto,
edita **únicamente** las dos constantes al inicio de `zipf.py` (las importan
`idiomas.py` y `plots.py`):

```python
RUTA_COUNTS = "./gutenberg_replica/data/counts"            # carpeta con los PG*_counts.txt
RUTA_META   = "./gutenberg_replica/metadata/metadata.csv"  # metadatos del SPGC
```

## Reproducir los resultados

Los scripts se ejecutan en orden y escriben todo en `outputs/` (se crea solo):

```bash
python zipf.py      # 1. ajuste + test KS de todo el corpus  -> outputs/corpus_analysis_full.csv
python idiomas.py   # 2. métricas por texto e idioma         -> outputs/metricas_texto.csv (+ tablas)
python plots.py     # 3. cuadros y figuras del TFG           -> outputs/*.tex, outputs/*.png
```

Los pasos 1 y 2 recorren todos los conteos con Monte Carlo y tardan (orden de horas).
El paso 1 usa una semilla fija (`seed=0`), por lo que los p-valores son reproducibles;
además `zipf.py` reanuda si se interrumpe. **Puedes saltarte 1 y 2 y ejecutar solo
`python plots.py`** para regenerar cuadros y figuras en segundos, utilizando la copia de
`metricas_texto.csv` incluida en el repositorio (en `outputs/`).

## Ficheros

| Fichero        | Contenido                                                          |
|----------------|-------------------------------------------------------------------|
| `zipf.py`      | Ajuste MLE de f1/f2/f3 y bondad de ajuste KS por Monte Carlo.      |
| `idiomas.py`   | Aceptación por idioma, fracción de *hapax* y razón de verosimilitud f1/f2 por texto. |
| `plots.py`     | Cuadros (LaTeX/CSV) y figuras del análisis multilingüe.           |

## Atribución

Los textos proceden de Project Gutenberg (dominio público). Su versión estandarizada
es el SPGC de M. Gerlach y F. Font-Clos, *A Standardized Project Gutenberg Corpus for
Statistical Analysis of Natural Language and Quantitative Linguistics*, Entropy 22(1):126
(2020), DOI [10.5281/zenodo.2422561](https://doi.org/10.5281/zenodo.2422561).
