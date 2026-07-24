"""Tablas y figuras del analisis multilingue de la ley de Zipf (extension del TFG).

Todo depende de un unico CSV por-texto (metricas_texto.csv). Ejecutar el modulo
(`python plots.py`) genera todas las tablas y figuras en outputs/. La figura de
ajuste individual (figure4) necesita ademas los PG*_counts.txt del corpus.

Convenios comunes:
  LANGS   idiomas con n>150 (zh excluido por artefacto de tokenizacion, la por outlier).
  ACC_THR aceptacion = p_f >= 0.05 (umbral del articulo).
  color   turbo ordenado por fraccion de hapax (gradiente morfologico).
  favor_f1 = fraccion de textos con preferencia SIGNIFICATIVA por f1 frente a f2 (z > 1.96).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from zipf import load_counts, make_dists, fit_beta, RUTA_COUNTS

# ============================================================ constantes
LANGS = ["en", "fr", "fi", "de", "nl", "it", "es", "pt", "el", "sv", "hu"]
VUONG_Z = 1.96      # |z| > 1.96  <=>  p_LR < 0.05 (Moreno-Sanchez et al., Tabla 2)
NOMBRES = {"en": "Inglés", "fr": "Francés", "nl": "Neerlandés", "es": "Español",
           "it": "Italiano", "sv": "Sueco", "de": "Alemán", "pt": "Portugués",
           "el": "Griego", "fi": "Finés", "hu": "Húngaro"}
C1_COLS = ["lengua", "cod", "n", "L", "N", "ttr", "hapax"]
C2_COLS = ["lengua", "cod", "hapax", "n", "acc_f1", "acc_f2", "acc_f3",
           "favor_f1", "favor_f2", "ratio_f2_f1"]
DEC = {"n": 0, "L": 0, "N": 0, "ttr": 3, "hapax": 3, "acc_f1": 3,
       "acc_f2": 3, "acc_f3": 3, "favor_f1": 3, "favor_f2": 3, "ratio_f2_f1": 2}
GRADIENT = ["en", "fr", "es", "fi", "hu"]
ACC_THR = 0.05
CSV = "outputs/metricas_texto.csv"
HAPAX_AGG = "median"


# ============================================================ utilidades compartidas
def load(csv=CSV):
    return pd.read_csv(csv).dropna(subset=["lang"])


def hapax_lang(df, langs=LANGS, agg=HAPAX_AGG):
    return df[df.lang.isin(langs)].groupby("lang").hapax_frac.agg(agg)


def order_by_hapax(df, langs=LANGS):
    return list(hapax_lang(df, langs).sort_values().index)


def turbo(order):
    return dict(zip(order, plt.cm.turbo(np.linspace(0.05, 0.95, len(order)))))


def favor(llr_z, side=+1):
    # fraccion de textos con preferencia SIGNIFICATIVA: side=+1 -> f1, side=-1 -> f2
    return float((side * llr_z > VUONG_Z).mean())


# ============================================================ tablas (Cuadros 1 y 2)



def corpus_metrics(df, langs=LANGS, acc_thr=ACC_THR):
    d = df[df.lang.isin(langs)].copy()
    d["ttr"] = d.N / d.L                    # cociente por texto; la mediana se toma despues
    acc = lambda s: float((s >= acc_thr).mean())
    g = d.groupby("lang")
    t = pd.DataFrame({
        "n": g.size(), "L": g.L.median(), "N": g.N.median(), "ttr": g.ttr.median(),
        "hapax": g.hapax_frac.agg(HAPAX_AGG),        # Cuadro y figuras
        "acc_f1": g.p_f1.apply(acc), "acc_f2": g.p_f2.apply(acc),
        "acc_f3": g.p_f3.apply(acc),
        "favor_f1": g.llr_z.apply(favor),
        "favor_f2": g.llr_z.apply(lambda z: favor(z, -1)),
    })
    t["ratio_f2_f1"] = t.acc_f2 / t.acc_f1
    return (t.reset_index()
             .assign(cod=lambda x: x.lang, lengua=lambda x: x.lang.map(NOMBRES)))


def cuadro1(m):                              # tamano tipico, orden por hapax mediana
    return m.sort_values("hapax")[C1_COLS].reset_index(drop=True)


def cuadro2(m):                              # aceptacion y preferencia, orden por hapax mediana
    return m.sort_values("hapax")[C2_COLS].reset_index(drop=True)


def _num(x, dec):                            # coma decimal y separador de miles
    return f"{x:,.{dec}f}".replace(",", "@").replace(".", ",").replace("@", "\\,")


def to_latex(t, path):
    # cuerpo del entorno tabular: una fila por lengua, sin cabecera ni \hline
    filas = [" & ".join(str(r[c]) if c in ("lengua", "cod") else _num(r[c], DEC[c])
                        for c in t.columns) + r" \\" for _, r in t.iterrows()]
    open(path, "w", encoding="utf-8").write("\n".join(filas) + "\n")
    return filas


# ============================================================ figura: supervivencia de p-valores
def _surv(p, grid):
    p = p.dropna().values
    return np.array([(p >= t).mean() for t in grid])


def fig_survival(df, langs=LANGS, funcs=("f1", "f2"), style="gradient",
                 highlight=GRADIENT, out=None):
    d = df[df.lang.isin(langs)]
    order = order_by_hapax(d, langs)
    hp, color = hapax_lang(d, langs), turbo(order)
    grid = np.linspace(0.01, 1, 100)
    ymax = max(_surv(d.loc[d.lang == l, f"p_{f}"], grid).max()
               for l in order for f in funcs) * 1.05

    fig, axes = plt.subplots(1, len(funcs), figsize=(5.2 * len(funcs), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, funcs):
        for l in order:
            y = _surv(d.loc[d.lang == l, f"p_{f}"], grid)
            if style == "highlight" and l not in highlight:
                ax.plot(grid, y, color="0.82", lw=1.0, zorder=1)
            else:
                ax.plot(grid, y, color=color[l], lw=1.8 if style == "highlight" else 1.4, zorder=3)
        ax.axvline(ACC_THR, ls="--", color="0.5", lw=0.8, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
        ax.set_xlabel("p"); ax.set_title(f, fontsize=10)
    axes[0].set_ylabel(r"$S(p)$ = fraccion con p-valor $\geq p$")

    shown = order if style == "gradient" else [l for l in order if l in highlight]
    handles = [Line2D([], [], color=color[l], lw=2, label=f"{l} ({hp[l]:.2f})") for l in shown]
    if style == "highlight":
        handles.append(Line2D([], [], color="0.82", lw=1.5, label="otros"))
    axes[-1].legend(handles=handles, title="idioma (hapax)", fontsize=7.5,
                    loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out or f"outputs/fig_survival_{style}.png", dpi=150, bbox_inches="tight")
    return fig


# ============================================================ figura: scatter hapax vs favor_f1
_expit = lambda t: 1 / (1 + np.exp(-t))
_sigmoid = lambda h, a, b: _expit(a + b * h)
_OFFSET = {"fi": (-14, 4), "hu": (6, -4), "el": (-6, 7), "sv": (6, -11),
           "de": (6, 3), "nl": (-16, -3), "es": (6, 3)}
_TITLES = {"all": "todos los textos",
           "acc_favor": "favor: aceptados; hapax: todos (art., Tabla 1)",
           "acc_both": "solo aceptados (ambos ejes)"}


def _scatter_stats(df, langs, mode):
    d = df[df.lang.isin(langs)]
    acc = d[(d.p_f1 >= ACC_THR) | (d.p_f2 >= ACC_THR)]
    hap_src = acc if mode == "acc_both" else d
    fav_src = acc if mode in ("acc_favor", "acc_both") else d
    g = fav_src.groupby("lang")
    s = pd.DataFrame({"n": g.size(), "hapax": hapax_lang(hap_src, langs),
                      "favor": g.llr_z.apply(favor)})
    s["favor_se"] = np.sqrt(s.favor * (1 - s.favor) / s.n)
    return s.reindex(langs)


def fig_scatter(df, langs=LANGS, mode="all", out=None):
    s = _scatter_stats(df, langs, mode)
    (a, b), _ = curve_fit(_sigmoid, s.hapax, s.favor, p0=[-17, 32], maxfev=10000)
    r2 = 1 - np.sum((s.favor - _sigmoid(s.hapax, a, b))**2) / np.sum((s.favor - s.favor.mean())**2)
    rho = spearmanr(s.hapax, s.favor).correlation

    fig, ax = plt.subplots(figsize=(6.2, 5))
    xs = np.linspace(s.hapax.min() - .03, s.hapax.max() + .03, 200)
    ax.axhline(1, color="0.9", lw=0.8, zorder=0)
    ax.plot(xs, _sigmoid(xs, a, b), color="0.2", lw=1.8, zorder=2, label="logistica")
    ax.errorbar(s.hapax, s.favor, yerr=s.favor_se, fmt="o", ms=6,
                color="steelblue", ecolor="0.7", capsize=2, lw=1, zorder=3)
    for l in s.index:
        ax.annotate(l, (s.loc[l, "hapax"], s.loc[l, "favor"]), textcoords="offset points",
                    xytext=_OFFSET.get(l, (5, 4)), fontsize=8.5, color="0.2")
    txt = (f"favor_f1 = sigma({b:.1f}*hapax - {abs(a):.1f})\n"
           f"punto medio (favor=0.5): hapax = {-a/b:.3f}\n"
           f"pseudo-$R^2$ = {r2:.2f}   Spearman $\\rho$ = {rho:.2f}")
    ax.annotate(txt, xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax.set_title(_TITLES[mode], fontsize=9, color="0.4")
    ax.set_xlabel("fraccion de hapax (mediana por idioma)")
    ax.set_ylabel("favor_f1 = fraccion de textos con f1 preferida (z>1.96)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out or f"outputs/fig_scatter_{mode}.png", dpi=150, bbox_inches="tight")
    return fig


# ============================================================ figura: distribucion de beta (box)
def fig_beta_box(df, langs=LANGS, func="f1", ref=2.0, out=None):
    d = df[df.lang.isin(langs)]
    order = order_by_hapax(d, langs)
    color = turbo(order)
    sel = d[d[f"p_{func}"] >= ACC_THR]
    data = [sel.loc[sel.lang == l, f"beta_{func}"].dropna().values for l in order]
    ns = [len(x) for x in data]
    allv = np.concatenate([x for x in data if len(x)])
    lo, hi = np.quantile(allv, [.01, .99]); pad = 0.1 * (hi - lo)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    if ref is not None:
        ax.axhline(ref, ls="--", color="0.5", lw=0.9, zorder=1)
        ax.annotate(f"Zipf clasico  beta = {ref:g}", xy=(0.99, ref),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color="0.4")
    bp = ax.boxplot(data, positions=range(1, len(order) + 1), patch_artist=True,
                    widths=0.62, showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, l in zip(bp["boxes"], order):
        patch.set_facecolor(color[l]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(order, ns)], fontsize=8)
    ax.set_xlabel("idioma (ordenado por hapax ->)")
    ax.set_ylabel(rf"$\beta_{{{func[-1]}}}$")
    ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(out or f"outputs/fig_beta_{func}_box.png", dpi=150, bbox_inches="tight")
    return fig


# ============================================================ figuras: efecto de L sobre beta
def _edges(L, per_decade):
    step = 1 / per_decade
    lo = np.floor(np.log10(L.min()) / step) * step
    hi = np.ceil(np.log10(L.max()) / step) * step
    return 10 ** np.arange(lo, hi + step, step)


def _binned(sub, col, edges, min_count):
    d = sub.assign(_bi=pd.cut(sub.L, edges, labels=False)).dropna(subset=["_bi"])
    g = d.groupby("_bi")[col]
    st = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
    st = st[st.n >= min_count]
    centers = np.sqrt(edges[:-1] * edges[1:])
    st["x"] = centers[st.index.astype(int)]
    return st


def fig_beta_en_f1f2(df, per_decade=4, min_count=5, ref=2.0, out="fig_beta_en_vs_L.png"):
    en = df[df.lang == "en"]
    edges = _edges(en[en.p_f1 >= ACC_THR].L, per_decade)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    if ref is not None:
        ax.axhline(ref, ls="--", color="0.6", lw=0.8)
    for func, c in [("f1", "C0"), ("f2", "C1")]:
        st = _binned(en[en[f"p_{func}"] >= ACC_THR], f"beta_{func}", edges, min_count)
        ax.plot(st.x, st["mean"], "-o", color=c, ms=4, lw=1.4, label=func)
        ax.fill_between(st.x, st["mean"] - st["std"], st["mean"] + st["std"], color=c, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("L (longitud del texto)"); ax.set_ylabel(r"$\beta$ medio")
    ax.set_title("ingles (banda = +/-1 std)", fontsize=9, color="0.4")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def fig_beta_langs_vs_L(df, func="f1", per_decade=2, min_accepted=150, min_count=25,
                        ref=2.0, out=None):
    d = df[df.lang.isin(LANGS)]
    sel = d[d[f"p_{func}"] >= ACC_THR]
    keep, hp = sel.groupby("lang").size(), hapax_lang(d)
    order = [l for l in hp.sort_values().index if keep.get(l, 0) > min_accepted]
    color = turbo(order)
    edges = _edges(sel[sel.lang.isin(order)].L, per_decade)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if ref is not None:
        ax.axhline(ref, ls="--", color="0.6", lw=0.8, zorder=1)
    for l in order:
        st = _binned(sel[sel.lang == l], f"beta_{func}", edges, min_count)
        ax.plot(st.x, st["mean"], "-o", color=color[l], ms=4, lw=1.4,
                label=f"{l} ({hp[l]:.2f})", zorder=3)
        ax.fill_between(st.x, st["mean"] - st["std"], st["mean"] + st["std"],
                        color=color[l], alpha=0.13, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("L (longitud del texto)"); ax.set_ylabel(rf"$\beta_{{{func[-1]}}}$ medio")
    ax.set_title(f"banda = +/-1 std; >={min_accepted} textos aceptados", fontsize=9, color="0.4")
    ax.legend(title="idioma (hapax)", fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(out or f"outputs/fig_beta_{func}_vs_L.png", dpi=150, bbox_inches="tight")
    return fig


def fig_beta_scatter(df, ref_lang="en", min_accepted=150, ref=2.0, out="outputs/fig_beta_f1_scatter.png"):
    d = df[df.lang.isin(LANGS)]
    sel = d[d.p_f1 >= ACC_THR]
    keep, hp = sel.groupby("lang").size(), hapax_lang(d)
    plot_langs = [l for l in hp.sort_values().index
                  if keep.get(l, 0) > min_accepted and l != ref_lang]
    color = turbo(plot_langs)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    def trend(s, **kw):
        m, b = np.polyfit(np.log10(s.L), s.beta_f1, 1)
        xs = np.array([s.L.min(), s.L.max()])
        ax.plot(xs, m * np.log10(xs) + b, **kw)

    if ref is not None:
        ax.axhline(ref, ls="--", color="0.7", lw=0.8, zorder=1)
    for l in plot_langs:
        s = sel[sel.lang == l]
        ax.scatter(s.L, s.beta_f1, s=7, color=color[l], alpha=0.22, edgecolors="none", zorder=2)
        trend(s, color=color[l], lw=2, zorder=4, label=f"{l} ({hp[l]:.2f})")
    r = sel[sel.lang == ref_lang]
    trend(r, color="0.3", lw=1.6, ls="--", zorder=4, label=f"{ref_lang} (ref., {hp[ref_lang]:.2f})")

    lo, hi = np.quantile(sel[sel.lang.isin(plot_langs)].beta_f1, [.01, .99])
    ax.set_xscale("log"); ax.set_ylim(lo - .1, hi + .1)
    ax.set_xlabel("L (longitud del texto)"); ax.set_ylabel(r"$\beta_1$")
    ax.legend(title="idioma (hapax)", fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


# ============================================================ Figura ajuste individual
def figura4_panels(df, langs=LANGS, funcs=("f1", "f2", "f3"), pmin=0.3):
    d = df[df.lang.isin(langs)] if langs else df
    col = {"f1": "C0", "f2": "C1", "f3": "C2"}
    panels = []
    for f in funcs:
        if f=="f3":
            p_min_final=0.3
        else:
            p_min_final=pmin
        r = d[d[f"p_{f}"] > p_min_final].sort_values("L").iloc[-1]
        panels.append((r.id, f, f"{r.lang}: {str(r.title)[:40]}  (L={int(r.L)})", col[f]))
    return panels


def _ccdf(data):
    vals, cnt = np.unique(data, return_counts=True)
    return vals, np.cumsum(cnt[::-1])[::-1] / data.size          # S_emp(n)=P(X>=n)


def _pmf_points(data, per_decade=5, threshold=10):
    # n < threshold: dato empirico por entero (pmf = conteo/N), en el entero exacto,
    #                igual que la survival. n >= threshold: densidad log-binned
    #                (masa/anchura, convencion del articulo). Error de Poisson.
    N, nmax = data.size, int(data.max())
    vals, cnt = np.unique(data, return_counts=True)
    lo = vals < threshold
    x, y, e = vals[lo].astype(float), cnt[lo] / N, np.sqrt(cnt[lo]) / N
    if nmax >= threshold:
        k = max(2, int(per_decade * np.log10(nmax + 1)) + 1)
        full = np.floor(np.logspace(0, np.log10(nmax + 1), k)).astype(int)
        edges = np.unique(np.concatenate([[threshold], full[full > threshold], [nmax + 1]]))
        c, _ = np.histogram(data, bins=edges)
        w, ctr, m = np.diff(edges), np.sqrt(edges[:-1] * edges[1:]), c > 0
        x = np.concatenate([x, ctr[m]])
        y = np.concatenate([y, (c / (N * w))[m]])
        e = np.concatenate([e, (np.sqrt(c) / (N * w))[m]])
    return x, y, e


def plot_fit(ax, data, dist, beta, color="C0", title=None):
    v, S = _ccdf(data)
    c, dens, err = _pmf_points(data)
    ax.loglog(v, S, "o", mfc="none", ms=4, color=color, label="survival")
    ax.errorbar(c, dens, err, fmt="o", ms=4, color=color, label="pmf (log-bin)")
    nn = np.unique(np.round(np.logspace(0, np.log10(data.max()), 300)).astype(int))
    ax.loglog(nn, dist.survival(nn, beta), color=color, lw=1)
    ax.loglog(nn, dist.pmf(nn, beta), color=color, lw=1, label="fit")

    x_vals = np.concatenate([v, c, nn])
    y_vals = np.concatenate([S, dens, dist.survival(nn, beta), dist.pmf(nn, beta)])
    x_vals = x_vals[np.isfinite(x_vals) & (x_vals > 0)]
    y_vals = y_vals[np.isfinite(y_vals) & (y_vals > 0)]

    if x_vals.size:
        ax.set_xlim(max(1, x_vals.min() / 10), max(x_vals.max() * 1.1, 10))
    if y_vals.size:
        ymin = max(y_vals.min() / 10, 1e-9)
        ymax = max(y_vals.max() * 1.1, 1.0)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("n"); ax.set_ylabel("prob.")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def figure4(panels, folder=".", out="outputs/figure4.png"):
    #panels: lista de (pg_id, nombre_dist, titulo, color). Un panel por texto
    d = make_dists()
    fig, axes = plt.subplots(len(panels), 1, figsize=(6, 4.3 * len(panels)))
    for ax, (pid, name, title, color) in zip(np.atleast_1d(axes), panels):
        n = load_counts(f"{folder}/{pid}_counts.txt")
        b = fit_beta(n, d[name])
        plot_fit(ax, n, d[name], b, color, f"{title}  ({name}, β={b:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


# ============================================================ generar todo
def generar_todo(df=None):
    df = load() if df is None else df
    m = corpus_metrics(df)
    for nombre, t in [("cuadro1", cuadro1(m)), ("cuadro2", cuadro2(m))]:
        t.to_csv(f"outputs/{nombre}.csv", index=False)
        to_latex(t, f"outputs/{nombre}.tex")
    for style in ("gradient", "highlight"):
        fig_survival(df, style=style)
    for mode in ("all", "acc_favor", "acc_both"):
        fig_scatter(df, mode=mode)
    fig_beta_box(df, func="f1", ref=2.0)
    fig_beta_box(df, func="f2", ref=2.0)
    fig_beta_en_f1f2(df)
    fig_beta_langs_vs_L(df)
    fig_beta_scatter(df)
    figure4(figura4_panels(df, langs=LANGS, funcs=("f1", "f2", "f3")), folder=RUTA_COUNTS, out="outputs/figura_individual.png")
    plt.close("all")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    generar_todo()
    print("tablas y figuras generadas en outputs/")
