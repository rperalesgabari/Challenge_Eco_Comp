# =====================================================
# 💼 APP DE INVERSIÓN — CARTERA ÓPTIMA SEGÚN EL PERFIL DE RIESGO
# Versión Streamlit (para share.streamlit.io)
# Materia: Economía Computacional · Challenge de IA aplicada
# =====================================================
# Esta app es la versión web del notebook Challenge_Final.ipynb.
# La LÓGICA (Markowitz, optimización, ranking, gráficos, PDF) es la MISMA del
# notebook: lo único que cambia es la interfaz, que ahora usa widgets de Streamlit
# en lugar de ipywidgets / preguntas por consola, y la descarga de datos se cachea
# para no bajar precios de Yahoo en cada interacción.
# =====================================================

import time
import io
from datetime import date

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib
matplotlib.use("Agg")                      # backend sin ventana (servidor)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

# yfinance es OPCIONAL: si no está / no hay internet, se usan datos SIMULADOS.
try:
    import yfinance as yf
    HAY_YFINANCE = True
except Exception:
    HAY_YFINANCE = False

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="App de Inversión — Markowitz", page_icon="💼", layout="wide")

# =====================================================
# 2) DATOS — PASO 1: BUSCAR EL UNIVERSO DE ACCIONES EN YAHOO FINANCE
# =====================================================
SCREENERS = {
    'conservador': 'undervalued_large_caps',      # grandes y estables
    'intermedio':  'undervalued_growth_stocks',   # valor + crecimiento
    'agresivo':    'aggressive_small_caps',       # crecimiento agresivo
}

# Lista de respaldo: solo se usa si NO hay internet / yfinance (para poder simular).
TICKERS_RESPALDO = ['AAPL', 'AVGO', 'QCOM', 'LMT', 'MSFT', 'GOOGL', 'AMZN', 'META', 'V', 'JPM',
                    'C', 'FXI', 'KO', 'IBM', 'CSCO', 'MCD', 'PG', 'UNH', 'JNJ', 'MRK', 'SBUX',
                    'RIO', 'VZ', 'GLD', 'SPY', 'LLY', 'TMUS', 'COST', 'GE', 'ABBV']


def _simbolos_de(resp):
    """Saca la lista de tickers de la respuesta del screener (tolera varias versiones de yfinance)."""
    quotes = None
    if isinstance(resp, dict):
        quotes = resp.get('quotes')
        if quotes is None:
            try:
                quotes = resp['finance']['result'][0]['quotes']
            except Exception:
                quotes = None
    if not quotes:
        return []
    return [q['symbol'] for q in quotes if isinstance(q, dict) and q.get('symbol')]


def buscar_tickers_yahoo(perfil, size=40):
    """Devuelve la lista de tickers que Yahoo Finance sugiere hoy para ese perfil."""
    screener = SCREENERS[perfil.lower()]
    for intento in (lambda: yf.screen(screener, size=size),
                    lambda: yf.screen(screener)):
        try:
            syms = _simbolos_de(intento())
            if syms:
                return syms[:size]
        except Exception:
            pass
    try:
        s = yf.Screener(); s.set_predefined_body(screener)
        syms = _simbolos_de(s.response)
        if syms:
            return syms[:size]
    except Exception:
        pass
    raise RuntimeError(f'No pude traer tickers de Yahoo para el perfil "{perfil}".')


def universo_comun(por_perfil=15):
    """Une las sugerencias de los 3 perfiles en un solo universo (sin repetidos)."""
    universo, vistos = [], set()
    for p in SCREENERS:
        try:
            for t in buscar_tickers_yahoo(p, size=por_perfil):
                if t not in vistos:
                    vistos.add(t); universo.append(t)
        except Exception:
            pass
    return universo


# =====================================================
# 2) DATOS — PASO 2: DESCARGAR (o SIMULAR) LOS PRECIOS
# =====================================================
START = '2020-01-01'
END   = '2025-05-16'


def descargar_precios(tickers, start, end):
    """Baja precios de cierre de Yahoo (de a 10) y se queda con los que tienen ≥60% de datos."""
    partes = {}
    for i in range(0, len(tickers), 10):
        grupo = tickers[i:i + 10]
        df = yf.download(grupo, start=start, end=end, progress=False)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=grupo[0])
        partes.update(df.to_dict('series'))
        time.sleep(1)
    precios = pd.DataFrame(partes)
    minimo  = int(len(precios) * 0.6)
    return precios.dropna(axis=1, thresh=minimo).dropna()


def simular_precios(tickers, start, end, seed=42):
    """Precios SINTÉTICOS con Movimiento Browniano Geométrico (GBM) correlacionado."""
    rng     = np.random.default_rng(seed)
    fechas  = pd.bdate_range(start, end)
    dias, n = len(fechas), len(tickers)
    mu  = rng.normal(0.08, 0.10, n) / 252
    vol = rng.uniform(0.15, 0.45, n) / np.sqrt(252)
    A    = rng.normal(0, 1, (n, n)); C = A @ A.T
    d    = np.sqrt(np.diag(C)); corr = C / np.outer(d, d)
    corr = 0.3 * corr + 0.7 * np.eye(n)
    z    = rng.standard_normal((dias, n)) @ np.linalg.cholesky(corr).T
    rets = mu + vol * z
    precios = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(precios, index=fechas, columns=tickers)


@st.cache_data(show_spinner=False)
def cargar_datos(forzar_simulado=False):
    """Arma el universo y trae los precios. Cacheado: solo corre una vez por sesión
       (o cuando cambian los parámetros). Devuelve (data, fuente)."""
    # 1) universo
    if HAY_YFINANCE and not forzar_simulado:
        try:
            tickers = universo_comun(por_perfil=15)
            if not tickers:
                tickers = list(TICKERS_RESPALDO)
        except Exception:
            tickers = list(TICKERS_RESPALDO)
    else:
        tickers = list(TICKERS_RESPALDO)

    # 2) precios
    fuente = 'reales'
    if HAY_YFINANCE and not forzar_simulado:
        try:
            data = descargar_precios(tickers, START, END)
            if data.empty or data.shape[1] < 3:
                raise RuntimeError('Yahoo devolvió pocos datos.')
        except Exception:
            data, fuente = simular_precios(tickers, START, END), 'simulados'
    else:
        data, fuente = simular_precios(tickers, START, END), 'simulados'

    data = data.dropna(axis=1, how='all').dropna()
    return data, fuente


# =====================================================
# BARRA LATERAL (reemplaza las preguntas por consola / ipywidgets)
# =====================================================
st.sidebar.header("⚙️ Configuración")
capital = st.sidebar.number_input("¿Cuánto querés invertir? (US$)",
                                  min_value=100.0, value=10000.0, step=500.0, format="%.2f")
perfil_label = st.sidebar.radio("¿Qué tipo de inversor sos?",
                                ["Conservador (menos riesgo)",
                                 "Intermedio (equilibrado)",
                                 "Agresivo (más retorno)"])
perfil = {"Conservador (menos riesgo)": "conservador",
          "Intermedio (equilibrado)":   "intermedio",
          "Agresivo (más retorno)":     "agresivo"}[perfil_label]
n_acciones = st.sidebar.slider("¿Cuántas acciones querés en la cartera?", 3, 15, 8)
forzar_sim = st.sidebar.checkbox("Usar datos simulados (demo sin internet)",
                                 value=not HAY_YFINANCE,
                                 help="Útil para una demo estable: no depende de Yahoo Finance.")
st.sidebar.caption("Material educativo. No constituye recomendación de inversión.")

# =====================================================
# CARGA DE DATOS Y CÁLCULOS BASE (idénticos al notebook)
# =====================================================
st.title("💼 App de Inversión — Cartera óptima por perfil de riesgo")
st.caption("Modelo de Markowitz · Python · Yahoo Finance · scipy.optimize")

with st.spinner("Cargando universo de acciones y precios..."):
    data, FUENTE_DATOS = cargar_datos(forzar_simulado=forzar_sim)

tickers = list(data.columns)
log_returns = np.log(data / data.shift()).dropna()
noa      = len(log_returns.columns)
mean_ret = log_returns.mean() * 252
cov      = log_returns.cov()  * 252

if FUENTE_DATOS == 'reales':
    st.success(f"Fuente de datos: **REALES (Yahoo Finance)** · {noa} activos · "
               f"{log_returns.index.min().date()} a {log_returns.index.max().date()}")
else:
    st.info(f"Fuente de datos: **SIMULADOS (GBM correlacionado)** · {noa} activos. "
            "Yahoo no estuvo disponible o elegiste el modo demo.")

# Métricas por activo (para ranking y Gráfico 2)
metricas = pd.DataFrame({
    'retorno':     log_returns.mean() * 252,
    'volatilidad': log_returns.std() * np.sqrt(252)
})
metricas['sharpe'] = metricas['retorno'] / metricas['volatilidad']


# =====================================================
# 4) LAS TRES CARTERAS (Markowitz) — funciones idénticas al notebook
# =====================================================
def port_ret(w):
    w = np.asarray(w, dtype=float)
    return float(np.sum(log_returns.mean() * w) * 252)

def port_vol(w):
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

def neg_sharpe(w):  return -port_ret(w) / port_vol(w)
def neg_ret(w):     return -port_ret(w)

w_max = max(0.30, 1.0 / noa)
bnds  = tuple((0.0, w_max) for _ in range(noa))
cons  = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
w0    = np.array(noa * [1.0 / noa])

def optimizar(funcion):
    r = sco.minimize(funcion, w0, method='SLSQP', bounds=bnds, constraints=cons)
    return r.x

PESOS = {
    'conservador': optimizar(port_vol),
    'intermedio':  optimizar(neg_sharpe),
    'agresivo':    optimizar(neg_ret),
}

estilos = {'conservador': ('o', '#2563eb'),
           'intermedio':  ('*', '#16a34a'),
           'agresivo':    ('^', '#dc2626')}


# =====================================================
# 5) RANKING DE ACCIONES POR PERFIL — idéntico al notebook
# =====================================================
def ranking(perfil):
    if perfil == 'conservador':
        return metricas.sort_values('volatilidad', ascending=True)
    if perfil == 'agresivo':
        return metricas.sort_values('retorno', ascending=False)
    return metricas.sort_values('sharpe', ascending=False)

def mejores_tickers(perfil, n=10):
    return list(ranking(perfil).head(n).index)


# =====================================================
# 5.1 + 9) CARTERA CON LAS MEJORES ACCIONES + FIGURAS — idéntico al notebook
# =====================================================
def cartera_detallada(perfil, n=8, capital=10000):
    perfil = perfil.lower()
    sel = mejores_tickers(perfil, n)
    sub = log_returns[sel]
    S   = sub.cov() * 252
    k   = len(sel)
    def ret(w): w = np.asarray(w, float); return float(np.sum(sub.mean() * w) * 252)
    def vol(w): w = np.asarray(w, float); return float(np.sqrt(w @ S.values @ w))
    if   perfil == 'conservador': objetivo = vol
    elif perfil == 'agresivo':    objetivo = lambda w: -ret(w)
    else:                         objetivo = lambda w: -ret(w) / vol(w)
    wmax = max(0.30, 1.0 / k)
    r = sco.minimize(objetivo, np.array(k * [1.0 / k]), method='SLSQP',
                     bounds=tuple((0.0, wmax) for _ in range(k)),
                     constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
    w       = r.x
    precios = data.iloc[-1]
    filas = []
    for t, p in pd.Series(w, index=sel).sort_values(ascending=False).items():
        if p > 0.005:
            costo = capital * p
            filas.append({'Acción': t,
                          '% capital':    p * 100,
                          'Costo (US$)':  costo,
                          'Precio (US$)': float(precios[t]),
                          'Nominales':    costo / float(precios[t])})
    tabla = pd.DataFrame(filas)
    metr  = {'retorno': ret(w), 'vol': vol(w), 'sharpe': ret(w) / vol(w),
             'total': float(tabla['Costo (US$)'].sum())}
    return tabla, metr


def fig_frontera():
    np.random.seed(0)
    M = 2500
    rs = np.zeros(M); vs = np.zeros(M)
    for i in range(M):
        w = np.random.random(noa); w /= w.sum()
        rs[i] = port_ret(w); vs[i] = port_vol(w)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(vs, rs, c=rs / vs, cmap='viridis', s=8, alpha=0.5)
    fig.colorbar(sc, label='Ratio de Sharpe')
    for perfil_, w in PESOS.items():
        m, c = estilos[perfil_]
        ax.scatter(port_vol(w), port_ret(w), marker=m, s=320, c=c,
                   edgecolors='black', linewidths=1.5, label=perfil_.capitalize(), zorder=5)
    ax.set_xlabel('Riesgo (volatilidad anual)'); ax.set_ylabel('Retorno esperado anual')
    ax.set_title(f'Frontera eficiente de Markowitz  (datos {FUENTE_DATOS})')
    ax.legend(); fig.tight_layout()
    return fig


def fig_riesgo_retorno():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(metricas['volatilidad'], metricas['retorno'], c=metricas['sharpe'],
                    cmap='plasma', s=80, edgecolors='black', linewidths=0.5)
    fig.colorbar(sc, label='Ratio de Sharpe')
    for t in metricas.index:
        ax.annotate(t, (metricas.loc[t, 'volatilidad'], metricas.loc[t, 'retorno']),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Riesgo (volatilidad anual)'); ax.set_ylabel('Retorno esperado anual')
    ax.set_title(f'Riesgo vs retorno por acción  (datos {FUENTE_DATOS})')
    fig.tight_layout()
    return fig


def fig_pesos():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (perfil_, w) in zip(axes, PESOS.items()):
        s = pd.Series(w, index=mean_ret.index)
        s = s[s > 0.005].sort_values(ascending=False)
        ax.bar(range(len(s)), s.values * 100, color=estilos[perfil_][1])
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s.index, rotation=45, ha='right', fontsize=8)
        ax.set_title(perfil_.capitalize()); ax.set_ylabel('% del capital')
    fig.suptitle(f'Reparto del capital por perfil  (datos {FUENTE_DATOS})')
    fig.tight_layout()
    return fig


def fig_tabla(perfil, tabla, metr, capital):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('Cartera de inversión sugerida', fontsize=18, fontweight='bold', y=0.96)
    fig.text(0.5, 0.915, f'Perfil: {perfil.capitalize()}    ·    '
             f'Capital: US$ {capital:,.2f}    ·    {date.today().isoformat()}',
             ha='center', fontsize=11)
    fig.text(0.5, 0.89, f'Fuente de datos: {FUENTE_DATOS}', ha='center', fontsize=9, color='gray')
    fig.text(0.5, 0.855, f'Retorno esperado anual: {metr["retorno"]*100:.1f}%       '
             f'Riesgo: {metr["vol"]*100:.1f}%       Sharpe: {metr["sharpe"]:.2f}',
             ha='center', fontsize=10)
    ax = fig.add_axes([0.07, 0.42, 0.86, 0.38]); ax.axis('off')
    vis = tabla.copy()
    vis['% capital']    = vis['% capital'].map(lambda x: f'{x:.1f}%')
    vis['Costo (US$)']  = vis['Costo (US$)'].map(lambda x: f'{x:,.2f}')
    vis['Precio (US$)'] = vis['Precio (US$)'].map(lambda x: f'{x:,.2f}')
    vis['Nominales']    = vis['Nominales'].map(lambda x: f'{x:,.4f}')
    t = ax.table(cellText=vis.values, colLabels=list(vis.columns),
                 cellLoc='center', loc='upper center')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
    for j in range(len(vis.columns)):
        c = t[0, j]; c.set_facecolor(estilos[perfil][1])
        c.set_text_props(color='white', fontweight='bold')
    fig.text(0.5, 0.37, f'Total asignado: US$ {metr["total"]:,.2f}',
             ha='center', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.04, 'Generado con la App de Inversión — Modelo de Markowitz.  '
             'Material educativo: no constituye recomendación de inversión.',
             ha='center', fontsize=8, color='gray')
    return fig


def construir_pdf(perfil, n, capital):
    """Arma el PDF de 4 páginas en memoria y devuelve los bytes (para descargar)."""
    tabla, metr = cartera_detallada(perfil, n, capital)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fabricar in (lambda: fig_tabla(perfil, tabla, metr, capital),
                         fig_frontera, fig_riesgo_retorno, fig_pesos):
            fig = fabricar(); pdf.savefig(fig); plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =====================================================
# CUERPO PRINCIPAL
# =====================================================
tab1, tab2, tab3 = st.tabs(["📌 Tu cartera", "📊 Comparación de perfiles", "📈 Gráficos de análisis"])

with tab1:
    st.subheader(f"Cartera sugerida — perfil {perfil.capitalize()}")
    tabla, metr = cartera_detallada(perfil, n_acciones, capital)

    c1, c2, c3 = st.columns(3)
    c1.metric("Retorno esperado anual", f"{metr['retorno']*100:.1f}%")
    c2.metric("Riesgo / volatilidad",   f"{metr['vol']*100:.1f}%")
    c3.metric("Ratio de Sharpe",        f"{metr['sharpe']:.2f}")

    vis = tabla.copy()
    vis['% capital']    = vis['% capital'].map(lambda x: f'{x:.1f}%')
    vis['Costo (US$)']  = vis['Costo (US$)'].map(lambda x: f'{x:,.2f}')
    vis['Precio (US$)'] = vis['Precio (US$)'].map(lambda x: f'{x:,.2f}')
    vis['Nominales']    = vis['Nominales'].map(lambda x: f'{x:,.4f}')
    st.dataframe(vis, hide_index=True, use_container_width=True)
    st.markdown(f"**Total asignado: US$ {metr['total']:,.2f}**")

    pdf_bytes = construir_pdf(perfil, n_acciones, capital)
    st.download_button("⬇️ Descargar cartera en PDF (4 páginas)",
                       data=pdf_bytes,
                       file_name=f"cartera_{perfil}.pdf",
                       mime="application/pdf")

with tab2:
    st.subheader("Las tres carteras óptimas (todo el universo)")
    resumen = pd.DataFrame({
        perfil_: {
            'Retorno esperado (anual)': f'{port_ret(w) * 100:.1f}%',
            'Riesgo / volatilidad':     f'{port_vol(w) * 100:.1f}%',
            'Ratio de Sharpe':          f'{port_ret(w) / port_vol(w):.2f}',
            'N° de acciones':           int((w > 0.005).sum()),
        }
        for perfil_, w in PESOS.items()
    }).T
    st.dataframe(resumen, use_container_width=True)
    st.caption("A medida que se pasa de conservador a agresivo, suben tanto el retorno "
               "esperado como el riesgo: ese es el trade-off central de Markowitz.")

with tab3:
    st.subheader("Gráfico 1 — Frontera eficiente")
    st.pyplot(fig_frontera())
    st.subheader("Gráfico 2 — Riesgo vs. retorno por acción")
    st.pyplot(fig_riesgo_retorno())
    st.subheader("Gráfico 3 — Pesos de la cartera por perfil")
    st.pyplot(fig_pesos())
