import os
import re
import json
import time
import math
import base64
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlencode, urlparse

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="Panel RDT - Capataces",
    page_icon="",
    layout="wide"
)

# Theme colors (matching dark theme)
THEME = {
    "primaryColor": "#0f766e",
    "backgroundColor": "#0b1320",
    "secondaryBackgroundColor": "#111827",
    "textColor": "#e5e7eb"
}

# ---------------------------
# Authentication Helper
# ---------------------------

def get_base_url() -> str | None:
    """
    Returns the base server URL (scheme + host) derived from the first non-empty
    OData endpoint configured via environment variables. If none are found,
    returns None. This base is used to construct the login URL for session
    token retrieval (POST /v1/sessions).
    """
    for env_var in ["ODATA_GANADERIA", "ODATA_FRUTALES", "ODATA_RIEGO"]:
        url = os.environ.get(env_var)
        if url:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
    return None


def login_and_get_token(email: str, password: str) -> str | None:
    """
    Given an ODK Central user email and password, attempts to authenticate
    against the server's /v1/sessions endpoint. On success, returns the
    bearer token string. On failure, returns None.
    """
    base = get_base_url()
    if not base:
        return None
    login_url = f"{base}/v1/sessions"
    try:
        resp = requests.post(
            login_url,
            json={"email": email, "password": password},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("token")
    except Exception:
        pass
    return None


# Persist token in session state; load from env if provided
if "token" not in st.session_state:
    st.session_state["token"] = os.environ.get("ODATA_TOKEN", "")

# If there is no token yet, display a login form and stop
if not st.session_state["token"]:
    st.title("Iniciar sesión")
    st.write("Ingresá tus credenciales de ODK Central para obtener un token de sesión.")
    with st.form("login_form"):
        email_input = st.text_input(
            "Email de usuario (ODK)",
            value=os.environ.get("ODATA_USER", ""),
            help="Ingresá tu email de usuario registrado en ODK Central.",
        )
        pwd_input = st.text_input(
            "Contraseña (ODK)",
            type="password",
            value=os.environ.get("ODATA_PASSWORD", ""),
            help="Ingresá tu contraseña de ODK Central.",
        )
        submitted = st.form_submit_button("Entrar")
    if submitted:
        token = login_and_get_token(email_input, pwd_input)
        if token:
            st.session_state["token"] = token
            st.success("Login exitoso: token obtenido.")
        else:
            st.error("Credenciales incorrectas o sin permisos. Reintentá.")
    # halt the script until token is obtained
    st.stop()

# Use the session token for all API calls
TOKEN = st.session_state["token"].strip()


# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def fetch_odata(url: str, bearer_token: str, top: int | None = None) -> pd.DataFrame:
    """
    Fetch an OData endpoint and follow @odata.nextLink pagination if present.
    Always authenticates using the provided bearer token.
    Returns a pandas DataFrame.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}" if bearer_token else "",
    }
    params = {}
    if top is not None:
        params["$top"] = str(top)
    data_frames = []
    next_url = url
    while next_url:
        resp = requests.get(next_url, headers=headers, params=params if url == next_url else None, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Error {resp.status_code} al consultar {next_url}: {resp.text[:300]}")
        payload = resp.json()
        if isinstance(payload, dict) and "value" in payload and isinstance(payload["value"], list):
            batch = payload["value"]
        elif isinstance(payload, list):
            batch = payload
        else:
            batch = payload.get("d", {}).get("results", [])
            if not isinstance(batch, list):
                batch = []
        if batch:
            df = pd.json_normalize(batch, sep=".")
            data_frames.append(df)
        next_url = payload.get("@odata.nextLink") or payload.get("odata.nextLink") or payload.get("d", {}).get("__next")
    if not data_frames:
        return pd.DataFrame()
    out = pd.concat(data_frames, ignore_index=True)
    # Try to parse date-like columns
    for col in out.columns:
        if re.search(r"(fecha|date|created|timestamp|time)$", col, flags=re.IGNORECASE):
            with pd.option_context("mode.chained_assignment", None):
                try:
                    out[col] = pd.to_datetime(out[col], errors="ignore", utc=True).dt.tz_convert("America/Argentina/Buenos_Aires")
                except Exception:
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
    out = [c for c in candidates if str(df[c].dtype).startswith("datetime")]
    return out or candidates


def render_hours_table(
    df: pd.DataFrame, empleado_col: str, fecha_col: str, periodo_inicio: pd.Timestamp
) -> pd.DataFrame:
    """Generate a pivoted hours table for the given period.

    The period spans from ``periodo_inicio`` (inclusive) to the 15th of the
    following month. The resulting DataFrame has one row per employee and one
    column per day in the period with missing values filled as ``0`` and
    columns ordered chronologically.
    """

    if (
        empleado_col not in df.columns
        or fecha_col not in df.columns
        or "horas" not in df.columns
    ):
        return pd.DataFrame()

    start = pd.to_datetime(periodo_inicio)
    end = (start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

    mask = (pd.to_datetime(df[fecha_col]) >= start) & (
        pd.to_datetime(df[fecha_col]) <= end
    )
    df_period = df.loc[mask].copy()

    if df_period.empty:
        rango = pd.date_range(start, end, freq="D")
        empty = pd.DataFrame(index=df[empleado_col].dropna().unique())
        empty = empty.assign(**{d.date(): 0 for d in rango})
        return empty

    df_period[fecha_col] = pd.to_datetime(df_period[fecha_col]).dt.date

    tabla = df_period.pivot_table(
        index=empleado_col,
        columns=fecha_col,
        values="horas",
        aggfunc="sum",
        fill_value=0,
    )

    rango = pd.date_range(start, end, freq="D").date
    tabla = tabla.reindex(columns=rango, fill_value=0)
    tabla = tabla.sort_index(axis=1)
    return tabla


def kpi_card(label: str, value, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def section_header(title: str, icon: str = ""):
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


def filter_block(
    df: pd.DataFrame,
    date_col_suggested: list[str],
    person_col_suggested: list[str],
    key_prefix: str,
):
    st.markdown("###  Filtros")
    c1, c2, c3 = st.columns([1, 1, 2])
    date_col = c1.selectbox(
        "Columna de fecha",
        options=["(ninguna)"] + date_col_suggested,
        index=1 if date_col_suggested else 0,
        key=f"{key_prefix}_date_col",
    )
    date_col = None if date_col == "(ninguna)" else date_col
    person_col = c2.selectbox(
        "Columna de capataz / usuario",
        options=["(ninguna)"] + person_col_suggested,
        index=1 if person_col_suggested else 0,
        key=f"{key_prefix}_person_col",
    )
    person_col = None if person_col == "(ninguna)" else person_col
    if date_col and date_col in df.columns and str(df[date_col].dtype).startswith("datetime"):
        min_date = pd.to_datetime(df[date_col]).min().date()
        max_date = pd.to_datetime(df[date_col]).max().date()
        start, end = c3.date_input(
            "Rango de fechas",
            (min_date, max_date),
            key=f"{key_prefix}_date_range",
        )
        if start and end:
            mask = (pd.to_datetime(df[date_col]).dt.date >= start) & (
                pd.to_datetime(df[date_col]).dt.date <= end
            )
            df = df.loc[mask].copy()
    if person_col and person_col in df.columns:
        people = ["(todas)"] + sorted(df[person_col].dropna().astype(str).unique().tolist())
        who = st.multiselect(
            "Capataz / Usuario",
            people,
            default=["(todas)"],
            key=f"{key_prefix}_who",
        )
        if who and "(todas)" not in who:
            df = df[df[person_col].astype(str).isin(who)].copy()
    return df, date_col, person_col


def data_health(df: pd.DataFrame, date_col: str | None, person_col: str | None):
    st.markdown("###  Calidad de datos (rápida)")
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
        st.success("Sin problemas obvios")


def download_button(df: pd.DataFrame, label: str = "Descargar CSV", filename: str = "export.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def view_hours(
    df: pd.DataFrame,
    empleado_col: str = "empleado",
    fecha_col: str = "fecha",
    horas_col: str = "horas",
):
    """Muestra una tabla de control de horas y permite descargarla.

    Si faltan las columnas requeridas ``empleado``, ``fecha`` u ``horas``,
    se informa mediante ``st.warning``.
    """

    missing = [
        col for col in (empleado_col, fecha_col, horas_col) if col not in df.columns
    ]
    if missing:
        st.warning("Faltan columnas requeridas: " + ", ".join(missing))
        return

    df_local = df.rename(
        columns={
            empleado_col: "empleado",
            fecha_col: "fecha",
            horas_col: "horas",
        }
    ).copy()
    df_local["fecha"] = pd.to_datetime(df_local["fecha"], errors="coerce")

    selected_month = st.date_input(
        "Mes de referencia", pd.Timestamp.today().replace(day=1)
    )
    periodo_inicio = selected_month.replace(day=16) - pd.offsets.MonthBegin(1)

    horas_tabla = render_hours_table(df_local, "empleado", "fecha", periodo_inicio)
    if horas_tabla.empty:
        st.warning("No hay datos para el período seleccionado")
        return

    st.dataframe(horas_tabla, use_container_width=True)
    download_button(horas_tabla.reset_index(), filename="horas_export.csv")


def view_tab_for_endpoint(name: str, url: str, bearer: str, top_preview: int | None):
    st.markdown(f"## {name}")
    with st.spinner(f"Cargando {name}..."):
        df = fetch_odata(url, bearer, top_preview)
    if df.empty:
        st.info("No se obtuvieron datos (o el esquema no retornó filas). Revisá credenciales, permisos y filtros.")
        return
    date_guess = guess_date_columns(df)
    people_guess = guess_people_columns(df)
    df_filtered, date_col, person_col = filter_block(
        df, date_guess, people_guess, name
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        section_header("Resumen")
        draw_overview(df_filtered, date_col, person_col)
    with c2:
        data_health(df_filtered, date_col, person_col)
    with c3:
        download_button(df_filtered, filename=f"{name.replace(' ', '_').lower()}_export.csv")
    section_header("Gráficos", "")
    charts(df_filtered, date_col, person_col)
    section_header("Tabla", "")
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
with st.sidebar.expander("Fuentes (OData)"):
    for k in list(default_urls.keys()):
        default_urls[k] = st.text_input(k, value=default_urls[k])
with st.sidebar.expander("Avanzado"):
    top_preview = st.number_input("$top (límite opcional)", min_value=0, value=0, help="0 = sin límite, útil para probar rápido.")
    top_preview = None if top_preview == 0 else int(top_preview)
    st.caption("Podés guardar estos valores como variables de entorno en Render/railway: ODATA_GANADERIA, ODATA_FRUTALES, ODATA_RIEGO, ODATA_USER, ODATA_PASSWORD, ODATA_TOKEN")

# Title and caption
st.title(" Panel de Cargas RDT — Capataces")
st.caption("Visualizá, filtrá y descargá registros de tus formularios ODK (vía OData).")

# Control de horas
st.header("Control de horas")
hours_file = st.file_uploader("Dataset de horas (CSV)", type="csv")
if hours_file is not None:
    df_hours = pd.read_csv(hours_file)
    view_hours(df_hours)

# Main tabs
tabs = st.tabs(list(default_urls.keys()))
for tab, (name, url) in zip(tabs, default_urls.items()):
    with tab:
        if not url:
            st.info("Ingresá la URL OData en la barra lateral para comenzar.")
        else:
            if name == "RDT Frutales":
                sub_tabs = st.tabs(["Datos", "Control de horas"])
                with sub_tabs[0]:
                    view_tab_for_endpoint(name, url, TOKEN, top_preview)
                with sub_tabs[1]:
                    with st.spinner("Cargando datos..."):
                        df_hours = fetch_odata(url, TOKEN, top_preview)
                    if df_hours.empty:
                        st.info("No se obtuvieron datos.")
                    else:
                        empleado_opts = guess_people_columns(df_hours) or df_hours.columns.tolist()
                        fecha_opts = guess_date_columns(df_hours) or df_hours.columns.tolist()
                        horas_opts = df_hours.columns.tolist()
                        empleado_col = st.selectbox("Columna empleado", empleado_opts)
                        fecha_col = st.selectbox("Columna fecha", fecha_opts)
                        horas_col = st.selectbox(
                            "Columna horas",
                            horas_opts,
                            index=horas_opts.index("horas") if "horas" in horas_opts else 0,
                        )
                        view_hours(df_hours, empleado_col, fecha_col, horas_col)
            else:
                view_tab_for_endpoint(name, url, TOKEN, top_preview)