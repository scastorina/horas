# Panel RDT — Capataces (OData → Streamlit)

App de Streamlit para visualizar formularios ODK vía OData (RDT Ganadería, Frutales, Riego).

## Ejecutar localmente
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ODATA_GANADERIA="https://.../RDT%20Ganaderia.svc"
export ODATA_FRUTALES="https://.../RDT%20Frutales.svc"
export ODATA_RIEGO="https://.../RDT%20Riego.svc"
# Opcional:
export ODATA_USER="usuario"
export ODATA_PASSWORD="password"
export ODATA_TOKEN="mi-token"  # si usás BEARER
streamlit run app.py
```

## Variables de entorno (Render/railway)
- `ODATA_GANADERIA`, `ODATA_FRUTALES`, `ODATA_RIEGO`
- `ODATA_USER`, `ODATA_PASSWORD` (si usás BASIC)
- `ODATA_TOKEN` (si usás BEARER)

## Notas
- La app sigue `@odata.nextLink` si hay paginación.
- Detecta columnas de fecha y de capataz/usuario por heurística (podés elegirlas desde la UI).
- Exporta CSV del filtro actual.

## Control de horas
- El dataset de horas debe contener las columnas `empleado`, `fecha` y `horas`.
- El período se calcula a partir del mes elegido con `periodo_inicio = selected_month.replace(day=16) - pd.offsets.MonthBegin(1)`.
- La tabla se actualiza automáticamente al cambiar el mes de referencia.