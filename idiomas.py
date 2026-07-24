"""
Analisis inter-idioma sobre el CSV definitivo del corpus.
  1. Tasas de aceptacion por idioma y cociente f2/f1   (desde el CSV).
  2. Mecanismo: fraccion de hapax/dis legomena por idioma (lee los counts).
  3. Distribucion del exponente beta por idioma          (desde el CSV).
La aceptacion sigue el paper: aceptar (no rechazar) si p >= ALPHA.
"""

import ast
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from zipf import load_counts, f1_pmf, f2_pmf, RUTA_COUNTS
from scipy.special import erfc

ALPHA = 0.05        # nivel de significacion
MIN_TEXTS = 30      # idiomas con menos textos no se tabulan
MIN_L = 100         # longitud minima del paper (L > 100 tokens)

# ========================================================================
# Carga y filtrado comun a los tres analisis:
#   - descarta los 10 fallos de ajuste (textos vacios, N = L = 0)
#   - descarta L <= 100 tokens (sin valor estadistico; Moreno-Sanchez 2016)
#   - idioma solo para textos monolingues; los multilingues se
#     descartan (lang = None) para no mezclar tokenizaciones distintas
# ========================================================================
def _single_lang(s):
    try:
        l = ast.literal_eval(s)
        return l[0] if len(l) == 1 else None
    except Exception:
        return None

def load_results(csv):
    df = pd.read_csv(csv).dropna(subset=["p_f1"])
    df = df[df.L > MIN_L]
    df["lang"] = df.language.map(_single_lang)
    return df


# ========================================================================
# 1. Aceptacion por idioma  ->  acc_f1, acc_f2, acc_f3, cociente f2/f1
# ========================================================================
def acceptance_by_language(df, alpha=ALPHA, min_texts=MIN_TEXTS):
    acc = lambda p: float((p >= alpha).mean())
    t = (df.dropna(subset=["lang"]).groupby("lang")
           .agg(n=("id", "size"), acc_f1=("p_f1", acc),
                acc_f2=("p_f2", acc), acc_f3=("p_f3", acc)))
    t = t[t.n >= min_texts].copy()
    t["ratio_f2_f1"] = t.acc_f2 / t.acc_f1          # inf si f1 nunca se acepta
    return t.sort_values("n", ascending=False)


# ========================================================================
# 2. Metricas por texto en una sola pasada de counts:
#    - hapax/dis: fraccion de tipos con n = 1 y n = 2 (el mecanismo).
#    - R: log-razon de verosimilitud f1 vs f2 con las betas ya ajustadas,
#         R = sum_i [log f1(n_i;b1) - log f2(n_i;b2)]  (R>0 favorece f1).
#         z: estadistico de Vuong (R normalizado); |z|>1.96 -> preferencia
#         significativa al 0.05, como la Tabla 2 de Moreno-Sanchez et al.
# ========================================================================
def _text_row(args):
    tid, path, b1, b2 = args
    n = load_counts(path).astype(float)
    d = np.log(f1_pmf(n, b1)) - np.log(np.clip(f2_pmf(n, b2), 1e-300, None))
    R = float(d.sum())
    s = float(d.std())
    z = R / (np.sqrt(n.size) * s) if s > 1e-12 else np.nan
    return {"id": tid,
            "hapax_frac": float(np.count_nonzero(n == 1)) / n.size,
            "dis_frac":   float(np.count_nonzero(n == 2)) / n.size,
            "llr": R,
            "llr_z": z,
            "p_lr": float(erfc(abs(z) / np.sqrt(2))) if np.isfinite(z) else np.nan}
 
def text_metrics(df, counts_dir, parallel=True):
    args = [(tid, os.path.join(counts_dir, f"{tid}_counts.txt"), b1, b2)
            for tid, b1, b2 in zip(df.id, df.beta_f1, df.beta_f2)]
    if parallel:
        with ProcessPoolExecutor() as ex:
            rows = list(ex.map(_text_row, args, chunksize=64))
    else:
        rows = [_text_row(a) for a in args]
    return df.merge(pd.DataFrame(rows), on="id")
 
def metrics_by_language(df_m, min_texts=MIN_TEXTS):
    fav = lambda s: lambda z: float((s * z > 1.96).mean())
    favors_f1 = lambda z: float((z > 0).mean())          # fraccion con R>0
    t = (df_m.dropna(subset=["lang"]).groupby("lang")
             .agg(n=("id", "size"),
                  hapax=("hapax_frac", "mean"),
                  dis=("dis_frac", "mean"),
                  favor_f1=("llr_z", fav(+1)),
                  favor_f2=("llr_z", fav(-1))))
    return t[t.n >= min_texts].sort_values("hapax", ascending=False)


# ========================================================================
# 3. Distribucion de beta por idioma (solo textos aceptados por esa dist.)
# ========================================================================
def beta_by_language(df, dist="f1", alpha=ALPHA, min_texts=MIN_TEXTS):
    b, p = f"beta_{dist}", f"p_{dist}"
    sel = df[df[p] >= alpha].dropna(subset=["lang"])
    t = (sel.groupby("lang")
            .agg(n=(b, "size"), mean=(b, "mean"),
                 std=(b, "std"), median=(b, "median")))
    return t[t.n >= min_texts].sort_values("n", ascending=False)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    df = load_results("outputs/corpus_analysis_full.csv")

    acc = acceptance_by_language(df)
    acc.to_csv("outputs/aceptacion_idioma.csv")
    print(acc.round(3), "\n")

    for d in ("f1", "f2"):
        betas = beta_by_language(df, d)
        betas.to_csv(f"outputs/beta_{d}_idioma.csv")
        print(f"beta {d}:"); print(beta_by_language(df, d).round(3), "\n")
    # Analisis 2 (lento: lee todos los counts). La ruta es RUTA_COUNTS, en zipf.py.
    df_m = text_metrics(df, RUTA_COUNTS)
    df_m.to_csv("outputs/metricas_texto.csv", index=False)
    metrics_by_language(df_m).to_csv("outputs/metrics_idioma.csv")
    print(metrics_by_language(df_m).round(4))
