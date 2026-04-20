# Forecasting de Ventas con Simulación Comercial

> Proyecto de analítica predictiva enfocado en estimar demanda diaria y evaluar escenarios de pricing para apoyar decisiones comerciales.

## Problema de Negocio

En contextos comerciales con múltiples productos, promociones y presión competitiva, anticipar la demanda es clave para planificar stock, ajustar descuentos y proyectar ingresos. Este proyecto aborda ese desafío mediante un enfoque de forecasting aplicado a ventas, incorporando además variables de precio y competencia.

## Objetivo del Proyecto

Desarrollar una solución que permita proyectar ventas diarias por producto para noviembre de 2025 y simular cómo cambian las unidades e ingresos esperados frente a distintos niveles de descuento y escenarios de precios de la competencia.

## Herramientas y Tecnologías

| Categoría | Herramientas |
| --- | --- |
| Lenguaje | `Python` |
| Análisis y manipulación | `pandas`, `NumPy` |
| Modelado | `scikit-learn` |
| Visualización | `Matplotlib`, `Seaborn` |
| Interfaz | `Streamlit` |
| Exploración | `Jupyter Notebook` |

## Metodología de Trabajo

1. Consolidación de datos de ventas históricas, precios y referencia competitiva.
2. Preparación de features temporales y comerciales, incluyendo lags de ventas, media móvil, ratio de precio y eventos del calendario.
3. Entrenamiento de un modelo de `HistGradientBoostingRegressor` para estimar unidades vendidas.
4. Generación de predicciones recursivas día a día para noviembre de 2025.
5. Construcción de una app en Streamlit para simular cambios de descuento y variaciones en precios de la competencia.
6. Presentación de KPIs y visualizaciones orientadas a toma de decisiones.

## Principales Hallazgos y Resultados

- Se construyó un flujo de predicción diaria orientado a negocio, con foco en 24 productos y escenarios comerciales concretos.
- El proyecto integra variables de calendario relevantes para retail, como el efecto de Black Friday.
- La simulación permite comparar rápidamente el impacto de cambios de descuento y de competencia sobre unidades e ingresos proyectados.
- El dashboard entrega una salida accionable mediante KPIs, tabla diaria y comparativa de escenarios.

## Valor para una Empresa

- Mejora la planificación comercial con una estimación anticipada de la demanda.
- Ayuda a evaluar decisiones de pricing sin depender únicamente de intuición.
- Facilita conversaciones entre equipos de negocio, operaciones y datos a partir de un mismo modelo.
- Convierte un modelo predictivo en una herramienta interpretable y utilizable por usuarios no técnicos.

## Estructura de Carpetas

```text
ForecastingVentas/
|-- README.md
|-- app/
|   `-- app.py
|-- data/
|   |-- processed/
|   `-- raw/
|-- docs/
|   `-- README.md
|-- models/
|   `-- modelo_Final.joblib
|-- notebooks/
|   |-- entrenamiento.ipynb
|   `-- forecasting.ipynb
`-- requirements.txt
```

## Cómo Reproducir el Proyecto

1. Posicionarse en la carpeta del proyecto:

   ```powershell
   cd ForecastingVentas
   ```

2. Crear y activar un entorno virtual:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instalar dependencias:

   ```powershell
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:

   ```powershell
   streamlit run app/app.py
   ```

## Próximos Pasos

- Incorporar métricas formales de validación como `MAE`, `RMSE` y `MAPE`.
- Documentar supuestos de negocio y criterios de modelado con mayor detalle.
- Agregar comparación entre modelos y una etapa de backtesting.
- Publicar una versión desplegada para acceso directo desde GitHub o CV.
