import os
import re
import json
import time
import math
import base64
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlencode, urlparse, parse_qs

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="Panel RDT - Capataces",
    page_icon="📋",
    layout="wide"
)

THEME = {
    "primaryColor": "#0f766e",
    "backgroundColor": "#0b1320",
    "secondaryBackgroundColor": "#111827",
    "textColor": "#e5e7eb"
}

# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def fetch_odata(url: str, auth_mode: str, username: str = "", password: str = "", bearer_token: str = "", top: int | None = None) -> pd.DataFrame:
    """
    Fetch an OData endpoint and follow @odata.nextLink pagination if present.
    Returns a pandas DataFrame.
    """
    headers = {"Accept": "application/json"}
    auth = None

    if auth_mode == "BASIC" and username and password:
        auth = (username, password)
    elif auth_mode == "BEARER" and bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"

    # Allow $top override for quick testing
    params = {}
    if top is not None:
        params["$top"] = str(top)

    data_frames = []
    next_url = url

    while next_url:
        resp = requests.get(next_url, headers=headers, auth=auth, params=params if url == next_url else None, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Error {resp.status_code} al consultar {next_url}: {resp.text[:300]}")

        payload = resp.json()
        # Try common shapes: {"value":[...]}, or direct list
        if isinstance(payload, dict) and "value" in payload and isinstance(payload["value"], list):
            batch = payload["value"]
        elif isinstance(payload, list):
            batch = payload
        else:
            # Try other shapes (some OData impls wrap differently)
            batch = payload.get("d", {}).get("results", [])
            if not isinstance(batch, list):
                # give up gracefully
                batch = []

        if batch:
            df = pd.json_normalize(batch, sep=".")
            data_frames.append(df)

        # in many OData impls, the next link is in "@odata.nextLink" or "odata.nextLink" or "d.__next"
        next_url = payload.get("@odata.nextLink") or payload.get("odata.nextLink") or payload.get("d", {}).get("__next")

    if not data_frames:
        return pd.DataFrame()

    out = pd.concat(data_frames, ignore_index=True)
    # Try to parse common date-like columns to datetime
    for col in out.columns:
        if re.search(r"(fecha|date|created|timestamp|time)$", col, flags=re.IGNORECASE):
            with pd.option_context("mode.chained_assignment", None):
                try:
                    out[col] = pd.to_datetime(out[col], errors="ignore", utc=True).dt.tz_convert("America/Argentina/Buenos_Aires")
                except Exception:
                    # try parsing naive
                    try:
                        out[col] = pd.to_datetime(out[col], errors="ignore")
                    except Exception:
                        pass
    return out


def guess_people_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for c in df.columns:
        if re.search(r"(capat[aá]z|operador|usuario|user|responsable|encargado|registr[oó]|createdBy|author)$", c, flags=re.IGNORECASE):
            candidates.append(c)
    return candidates


def guess_date_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for c in df.columns:
        if re.search(r"(fecha|date|día|dia|created|timestamp|time)$", c, flags=re.IGNORECASE):
            candidates.append(c)
    # Keep only datetime dtype if available
    out = [c for c in candidates if str(df[c].dtype).startswith("datetime")]
    return out or candidates


def kpi_card(label: str, value, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def section_header(title: str, icon: str = "📋"):
    st.subheader(f"{icon} {title}")


def draw_overview(df: pd.DataFrame, date_col: str | None, person_col: str | None):
    c1, c2, c3, c4 = st.columns(4)
    total_reg = len(df)
    c1.metric("Registros", f"{total_reg:,}".replace(",", "."))
    uniq_people = df[person_col].nunique() if person_col and person_col in df.columns else None
    c2.metric("Personas únicas", f"{uniq_people:,}".replace(",", ".") if uniq_people is not None else "—")
    if date_col and date_col in df.columns and str(df[date_col].dtype).startswith("datetime"):
        last_time = pd.to_datetime(df[date_col]).max()
        c3.metric("Último registro", last_time.strftime("%Y-%m-%d %H:%M") if pd.notna(last_time) else "—")
        by_day = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().tail(14)
        c4.metric("Prom. 7 días", f"{by_day.tail(7).mean():.1f}" if not by_day.empty else "—")
    else:
        c3.metric("Último registro", "—")
        c4.metric("Prom. 7 días", "—")


def charts(df: pd.DataFrame, date_col: str | None, person_col: str | None):
    if date_col and date_col in df.columns and str(df[date_col].dtype).startswith("datetime"):
        by_day = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().reset_index(name="registros")
        st.line_chart(by_day.set_index("date") if "date" in by_day.columns else by_day.set_index(by_day.columns[0]))
    if person_col and person_col in df.columns:
        top_people = df[person_col].fillna("—").value_counts().head(15)
        st.bar_chart(top_people)


def filter_block(df: pd.DataFrame, date_col_suggested: list[str], person_col_suggested: list[str]):
    st.markdown("### 🔍 Filtros")
    c1, c2, c3 = st.columns([1,1,2])

    date_col = c1.selectbox("Columna de fecha", options=["(ninguna)"] + date_col_suggested, index=1 if date_col_suggested else 0)
    date_col = None if date_col == "(ninguna)" else date_col

    person_col = c2.selectbox("Columna de capataz / usuario", options=["(ninguna)"] + person_col_suggested, index=1 if person_col_suggested else 0)
    person_col = None if person_col == "(ninguna)" else person_col

    if date_col and date_col in df.columns and str(df[date_col].dtype).startswith("datetime"):
        min_date = pd.to_datetime(df[date_col]).min().date()
        max_date = pd.to_datetime(df[date_col]).max().date()
        start, end = c3.date_input("Rango de fechas", (min_date, max_date))
        if start and end:
            mask = (pd.to_datetime(df[date_col]).dt.date >= start) & (pd.to_datetime(df[date_col]).dt.date <= end)
            df = df.loc[mask].copy()

    if person_col and person_col in df.columns:
        people = ["(todas)"] + sorted(df[person_col].dropna().astype(str).unique().tolist())
        who = st.multiselect("Capataz / Usuario", people, default=["(todas)"])
        if who and "(todas)" not in who:
            df = df[df[person_col].astype(str).isin(who)].copy()

    return df, date_col, person_col


def data_health(df: pd.DataFrame, date_col: str | None, person_col: str | None):
    st.markdown("### 🧪 Calidad de datos (rápida)")
    issues = []
    if date_col and date_col in df.columns:
        if df[date_col].isna().mean() > 0.05:
            issues.append(f"- {df[date_col].isna().mean():.1%} de fechas vacías")
    if person_col and person_col in df.columns:
        if df[person_col].isna().mean() > 0.05:
            issues.append(f"- {df[person_col].isna().mean():.1%} de usuarios vacíos")
    if not len(df.columns):
        issues.append("- No hay columnas en el set de datos")
    if issues:
        st.warning("Problemas detectados:\n" + "\n".join(issues))
    else:
        st.success("Sin problemas obvios 👌")


def download_button(df: pd.DataFrame, label: str = "Descargar CSV", filename: str = "export.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def view_tab_for_endpoint(name: str, url: str, auth_mode: str, username: str, password: str, bearer: str, top_preview: int | None):
    st.markdown(f"## {name}")
    with st.spinner(f"Cargando {name}..."):
        df = fetch_odata(url, auth_mode, username, password, bearer, top_preview)
    if df.empty:
        st.info("No se obtuvieron datos (o el esquema no retornó filas). Revisá credenciales, permisos y filtros.")
        return

    date_guess = guess_date_columns(df)
    people_guess = guess_people_columns(df)

    df_filtered, date_col, person_col = filter_block(df, date_guess, people_guess)

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        section_header("Resumen")
        draw_overview(df_filtered, date_col, person_col)
    with c2:
        data_health(df_filtered, date_col, person_col)
    with c3:
        download_button(df_filtered, filename=f"{name.replace(' ','_').lower()}_export.csv")

    section_header("Gráficos", "📈")
    charts(df_filtered, date_col, person_col)

    section_header("Tabla", "🧾")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)


# ---------------------------
# Sidebar: Configuración
# ---------------------------
st.sidebar.title("⚙️ Configuración")

default_urls = {
    "RDT Ganadería": os.environ.get("ODATA_GANADERIA", ""),
    "RDT Frutales": os.environ.get("ODATA_FRUTALES", ""),
    "RDT Riego": os.environ.get("ODATA_RIEGO", ""),
}

with st.sidebar.expander("Credenciales / Acceso"):
    auth_mode = st.radio("Modo de autenticación", ["Ninguna", "BASIC", "BEARER"], horizontal=True, index=0)
    user = st.text_input("Usuario (BASIC)", value=os.environ.get("ODATA_USER", ""))
    pwd = st.text_input("Password (BASIC)", type="password", value=os.environ.get("ODATA_PASSWORD", ""))
    token = st.text_input("Token (BEARER)", type="password", value=os.environ.get("ODATA_TOKEN", ""))

with st.sidebar.expander("Fuentes (OData)"):
    for k in list(default_urls.keys()):
        default_urls[k] = st.text_input(k, value=default_urls[k])

with st.sidebar.expander("Avanzado"):
    top_preview = st.number_input("$top (límite opcional)", min_value=0, value=0, help="0 = sin límite, útil para probar rápido.")
    top_preview = None if top_preview == 0 else int(top_preview)
    st.caption("Podés guardar estos valores como variables de entorno en Render/railway: ODATA_GANADERIA, ODATA_FRUTALES, ODATA_RIEGO, ODATA_USER, ODATA_PASSWORD, ODATA_TOKEN")

st.title("📋 Panel de Cargas RDT — Capataces")
st.caption("Visualizá, filtrá y descargá registros de tus formularios ODK (vía OData).")

# ---------------------------
# Main Tabs
# ---------------------------
tabs = st.tabs(list(default_urls.keys()))

for tab, (name, url) in zip(tabs, default_urls.items()):
    with tab:
        if not url:
            st.info("Ingresá la URL OData en la barra lateral para comenzar.")
        else:
            view_tab_for_endpoint(name, url, auth_mode, user, pwd, token, top_preview)