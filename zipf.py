"""
Replica de Moreno-Sanchez, Font-Clos & Corral (2016): ajuste de las tres
distribuciones de Zipf (f1, f2, f3) a las frecuencias de palabras de textos de
Project Gutenberg, por maxima verosimilitud, con test de bondad de ajuste KS
mediante Monte Carlo.

Flujo:  load_counts -> make_dists -> fit_beta / gof -> analyze_corpus.
Todo el analisis usa el cutoff inferior A = 1 (las distribuciones viven en
n = 1, 2, 3, ...).  Convenio de la survival:  S(n) = P(X >= n),  S(1) = 1.
"""

import glob
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import zeta, gammaln
from scipy.optimize import minimize_scalar

A = 1

# Rutas del corpus SPGC. Editar aqui: idiomas.py y plots.py las importan de este modulo.
RUTA_COUNTS = "./gutenberg_replica/data/counts"            # directorio con los PG*_counts.txt
RUTA_META   = "./gutenberg_replica/metadata/metadata.csv"  # metadatos generados por el SPGC


# ========================================================================
# 1. Carga
# ========================================================================
def load_counts(path):
    """Lee 'palabra<tab>frecuencia' y devuelve el array de frecuencias {n_i},
    la muestra que se ajusta (cada tipo aporta una observacion = su frecuencia)."""
    with open(path, encoding="utf-8") as f:
        return np.array([int(l.rsplit("\t", 1)[1]) for l in f if l.strip()],
                        dtype=np.int64)


# ========================================================================
# 2. Distribuciones:  pmf f(n)  y  survival S(n)
#    f1 Zipf | f2 potencia en la survival | f3 Mandelbrot (funcion Beta)
# ========================================================================
def f1_pmf(n, b):  return np.asarray(n, float) ** (-b) / zeta(b, A)
def f1_surv(n, b): return zeta(b, n) / zeta(b, A)

def f2_surv(n, b): return (np.asarray(n, float) / A) ** (1 - b)
def f2_pmf(n, b):
    n = np.asarray(n, float)                 # f2(n)=S2(n)-S2(n+1); expm1 evita
    return -f2_surv(n, b) * np.expm1((1 - b) * np.log1p(1 / n))   # cancelar en cola

def _logS3(n, b):                            # log S3(n)=B(n+1-b,b-1)/B(A+1-b,b-1)
    n = np.asarray(n, float)                 # (el factor Gamma(b-1) se cancela)
    return (gammaln(n + 1 - b) - gammaln(n)) - (gammaln(A + 1 - b) - gammaln(A))
def f3_surv(n, b): return np.exp(_logS3(n, b))
def f3_pmf(n, b):                            # recurrencia Gamma: f3(n)=S3(n)*(b-1)/n
    n = np.asarray(n, float)                 # (identidad exacta, sin restas)
    return f3_surv(n, b) * (b - 1) / n


# ========================================================================
# 3. Muestreadores (para el Monte Carlo)
#    f2 por inversion exacta;  f1 y f3 por rechazo con propuesta f2.
# ========================================================================
def sample_f2(size, b, rng):
    u = 1 - rng.random(size)                                    # u en (0,1]
    return np.floor(np.minimum(A * u ** (1 / (1 - b)), 1e18)).astype(np.int64)

def _reject(size, b, target_pmf, C, rng):
    out, got = [], 0
    while got < size:
        prop = sample_f2(int((size - got) * C * 1.3) + 16, b, rng)
        keep = prop[rng.random(prop.size) <= target_pmf(prop, b) / (C * f2_pmf(prop, b))]
        out.append(keep); got += keep.size
    return np.concatenate(out)[:size]

def sample_f1(size, b, rng):
    return _reject(size, b, f1_pmf, float(f1_pmf(A, b) / f2_pmf(A, b)), rng)  # C max en n=A

def sample_f3(size, b, rng):
    g = np.arange(A, 10 ** 6)
    return _reject(size, b, f3_pmf, float(np.max(f3_pmf(g, b) / f2_pmf(g, b))) * 1.001, rng)


# Cada distribucion agrupa pmf, survival, muestreador y rango valido de beta.
Dist = namedtuple("Dist", "name pmf survival sample bounds")

def make_dists():
    e = 1e-9
    return {"f1": Dist("f1", f1_pmf, f1_surv, sample_f1, (1 + e, 4)),
            "f2": Dist("f2", f2_pmf, f2_surv, sample_f2, (1 + e, 4)),
            "f3": Dist("f3", f3_pmf, f3_surv, sample_f3, (1 + e, 2 - e))}


# ========================================================================
# 4. Ajuste MLE:  beta que maximiza  sum_i log f(n_i)
# ========================================================================
def fit_beta(data, dist):
    vals, mult = np.unique(data, return_counts=True)
    def neg_ll(b):
        p = dist.pmf(vals, b)
        return np.inf if np.any(p <= 0) else -float(mult @ np.log(p))
    return float(minimize_scalar(neg_ll, bounds=dist.bounds, method="bounded").x)


# ========================================================================
# 5. Bondad de ajuste:  KS discreta + p-valor por Monte Carlo (Clauset 2009)
# ========================================================================
def ks_distance(data, dist, b):
    """d = max_n |S_emp(n) - S(n)|. El supremo cae en un valor observado o su +1."""
    vals, cnt = np.unique(data, return_counts=True)
    tail = np.append(np.cumsum(cnt[::-1])[::-1], 0)        # tail[i] = #obs >= vals[i]
    cand = np.union1d(vals, vals + 1)
    S_emp = tail[np.searchsorted(vals, cand)] / data.size
    return float(np.max(np.abs(S_emp - dist.survival(cand, b))))

def gof(data, dist, n_sims=100, rng=None):
    """Devuelve (beta, KS, p). p = fraccion de muestras sinteticas del modelo
    ajustado (cada una reajustada) cuya KS iguala o supera la observada."""
    rng = np.random.default_rng() if rng is None else rng
    b = fit_beta(data, dist)
    d = ks_distance(data, dist, b)
    ge = 0
    for _ in range(n_sims):
        s = dist.sample(data.size, b, rng)
        if ks_distance(s, dist, fit_beta(s, dist)) >= d:
            ge += 1
    return b, d, ge / n_sims


# ========================================================================
# 6. Corpus completo -> DataFrame consultable (metadatos + L + betas + p-valores)
# ========================================================================
_META = ["id", "title", "author", "authoryearofbirth", "authoryearofdeath",
         "language", "downloads", "subjects", "type"]

def analyze_file(path, n_sims=100, rng=None):
    """Analiza un solo PG*_counts.txt y devuelve un dict con id, N, L, betas y p-valores
    (los metadatos se anaden despues, en el merge del __main__)."""
    print(f"Analizando {path} ...")
    n = load_counts(path)
    row = {"id": os.path.basename(path).replace("_counts.txt", ""),
           "N": int(n.size), "L": int(n.sum())}
    if n.size == 0:                                  # fichero degenerado: sin datos
        for name in ("f1", "f2", "f3"):
            row[f"beta_{name}"] = row[f"p_{name}"] = np.nan
        return row
    for name, dist in make_dists().items():
        b, _, p = gof(n, dist, n_sims, rng)
        row[f"beta_{name}"], row[f"p_{name}"] = b, p
    return row

def _afile(args):                            # envoltorio pickleable para el pool
    return analyze_file(*args)

def _safe_afile(args):                           # nunca propaga: NaN si algo falla
    try:
        return _afile(args)
    except Exception as e:
        pid = os.path.basename(args[0]).replace("_counts.txt", "")
        row = {"id": pid, "N": np.nan, "L": np.nan}
        for name in ("f1", "f2", "f3"):
            row[f"beta_{name}"] = row[f"p_{name}"] = np.nan
        row["error"] = f"{type(e).__name__}: {e}"
        return row


def analyze_corpus(folder, out_csv, n_sims=100, seed=0, resume=True):
    """Analiza todos los PG*_counts.txt escribiendo cada fila a `out_csv` en
    cuanto termina. Si se interrumpe, relanzar reanuda saltando los id ya hechos."""
    paths = sorted(glob.glob(os.path.join(folder, "PG*_counts.txt")))
    seeds = np.random.SeedSequence(seed).spawn(len(paths))   # seed estable por ruta

    done = set()
    if resume and os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, usecols=["id"])["id"])
    args = [(p, n_sims, np.random.default_rng(s))
            for p, s in zip(paths, seeds)
            if os.path.basename(p).replace("_counts.txt", "") not in done]

    header = not os.path.exists(out_csv)
    with ProcessPoolExecutor() as ex:
        for fut in as_completed(ex.submit(_safe_afile, a) for a in args):
            pd.DataFrame([fut.result()]).to_csv(out_csv, mode="a",
                                                header=header, index=False)
            header = False

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    analyze_corpus(RUTA_COUNTS, "outputs/corpus_analysis.csv")
    df = pd.read_csv("outputs/corpus_analysis.csv").merge(
            pd.read_csv(RUTA_META, usecols=_META), on="id", how="left")
    df.to_csv("outputs/corpus_analysis_full.csv", index=False)