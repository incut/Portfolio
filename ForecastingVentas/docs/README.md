# Documentación Técnica — Forecasting de Ventas con Simulación Comercial

> Documento de referencia técnica del proyecto `ForecastingVentas`. Incluye el alcance del proyecto, diccionario de datos, metodología de modelado, métricas de evaluación y guía de uso de la aplicación.

---

## Índice

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Problema de Negocio](#problema-de-negocio)
3. [Objetivo del Proyecto](#objetivo-del-proyecto)
4. [Herramientas y Tecnologías](#herramientas-y-tecnologías)
5. [Estructura de Carpetas](#estructura-de-carpetas)
6. [Fuentes de Datos](#fuentes-de-datos)
7. [Diccionario de Datos](#diccionario-de-datos)
8. [Metodología de Trabajo](#metodología-de-trabajo)
9. [Ingeniería de Features](#ingeniería-de-features)
10. [Modelo Predictivo](#modelo-predictivo)
11. [Métricas de Evaluación](#métricas-de-evaluación)
12. [Predicción Recursiva](#predicción-recursiva)
13. [Aplicación Streamlit](#aplicación-streamlit)
14. [Cómo Reproducir el Proyecto](#cómo-reproducir-el-proyecto)
15. [Principales Hallazgos](#principales-hallazgos)
16. [Valor para una Empresa](#valor-para-una-empresa)

---

## Descripción del Proyecto

**ForecastingVentas** es un proyecto de analítica predictiva end-to-end orientado a la estimación de demanda diaria y la evaluación de escenarios comerciales de pricing. Cubre desde la preparación de los datos históricos hasta la exposición de predicciones interactivas a través de una aplicación web construida con Streamlit.

Demo online: [portfolio-forecasting-ventas.streamlit.app](https://portfolio-forecasting-ventas.streamlit.app)

---

## Problema de Negocio

En contextos comerciales con múltiples productos, promociones y presión competitiva, anticipar la demanda es clave para planificar stock, ajustar descuentos y proyectar ingresos. Este proyecto aborda ese desafío mediante un enfoque de forecasting aplicado a ventas, incorporando además variables de precio propio y de competencia.

---

## Objetivo del Proyecto

Desarrollar una solución que permita:

- Proyectar las ventas diarias por producto para **noviembre de 2025**.
- Simular cómo cambian las **unidades e ingresos esperados** frente a distintos niveles de descuento.
- Comparar tres **escenarios de precios de la competencia**: sin cambio, competencia baja precios 5%, competencia sube precios 5%.

---

## Herramientas y Tecnologías

| Categoría | Herramientas |
| --- | --- |
| Lenguaje | `Python 3` |
| Análisis y manipulación | `pandas`, `NumPy` |
| Modelado | `scikit-learn` (`HistGradientBoostingRegressor`) |
| Visualización | `Matplotlib`, `Seaborn` |
| Interfaz | `Streamlit` |
| Calendario de festivos | `holidays` |
| Exploración | `Jupyter Notebook` |
| Persistencia del modelo | `joblib` |

---

## Estructura de Carpetas

```text
ForecastingVentas/
|-- README.md
|-- app/
|   `-- app.py                          ← Aplicación Streamlit principal
|-- data/
|   |-- processed/
|   |   `-- inferencia_df_transformado.csv  ← Dataset de inferencia (noviembre 2025)
|   `-- raw/
|       |-- entrenamiento/
|       |   |-- ventas.csv              ← Ventas históricas (oct 2021 – dic 2024)
|       |   `-- competencia.csv         ← Precios de la competencia (mismo periodo)
|       `-- inferencia/
|           `-- ventas_2025_inferencia.csv  ← Base para construir el set de inferencia
|-- docs/
|   `-- README.md                       ← Este documento
|-- models/
|   `-- modelo_Final.joblib             ← Modelo entrenado y serializado
|-- notebooks/
|   |-- entrenamiento.ipynb             ← EDA, feature engineering y entrenamiento
|   `-- forecasting.ipynb               ← Preparación del set de inferencia
`-- requirements.txt
```

---

## Fuentes de Datos

### Dataset de Entrenamiento

| Archivo | Descripción | Registros | Periodo |
| --- | --- | --- | --- |
| `ventas.csv` | Ventas diarias por producto (24 SKUs) | 3 552 | oct 2021 – dic 2024 |
| `competencia.csv` | Precios diarios de Amazon, Decathlon y Deporvillage por producto | 3 552 | oct 2021 – dic 2024 |

Los dos archivos se unen por `fecha` + `producto_id` para construir el dataset de entrenamiento `df`.

### Dataset de Inferencia

| Archivo | Descripción |
| --- | --- |
| `ventas_2025_inferencia.csv` | Datos base de noviembre 2025 sin unidades reales, con precios y variables de contexto ya definidos |
| `inferencia_df_transformado.csv` | Dataset procesado y con todas las features listas para el modelo |

---

## Diccionario de Datos

### Variables originales (ventas + competencia)

| Variable | Tipo | Descripción |
| --- | --- | --- |
| `fecha` | `datetime` | Fecha del registro (granularidad diaria) |
| `producto_id` | `str` | Identificador único del producto (PROD_001 … PROD_024) |
| `nombre` | `str` | Nombre comercial del producto |
| `categoria` | `str` | Categoría principal: `Running`, `Fitness`, `Outdoor`, `Wellness` |
| `subcategoria` | `str` | Subcategoría del producto (16 valores únicos) |
| `precio_base` | `int` | Precio de catálogo sin descuento (€) |
| `es_estrella` | `bool` | `True` si el producto es considerado estrella de ventas |
| `unidades_vendidas` | `int` | Variable objetivo — unidades vendidas en ese día |
| `precio_venta` | `float` | Precio real de venta aplicado ese día (€) |
| `ingresos` | `float` | Ingresos generados = `precio_venta × unidades_vendidas` (€) |
| `Amazon` | `float` | Precio de la competencia Amazon ese día (€) |
| `Decathlon` | `float` | Precio de la competencia Decathlon ese día (€) |
| `Deporvillage` | `float` | Precio de la competencia Deporvillage ese día (€) |

### Variables derivadas (feature engineering)

| Variable | Tipo | Descripción |
| --- | --- | --- |
| `año` | `int` | Año extraído de `fecha` |
| `mes` | `int` | Mes (1–12) |
| `dia_mes` | `int` | Día del mes (1–31) |
| `dia_semana` | `int` | Día de la semana (0 = lunes, 6 = domingo) |
| `nombre_dia` | `str` | Nombre del día de la semana |
| `semana_año` | `int` | Número de semana ISO del año |
| `dia_año` | `int` | Día del año (1–366) |
| `trimestre` | `int` | Trimestre (1–4) |
| `semestre` | `int` | Semestre (1–2) |
| `es_fin_de_semana` | `bool` | `True` si es sábado o domingo |
| `es_laborable` | `bool` | `True` si es día laborable |
| `es_festivo` | `bool` | `True` si es festivo nacional (España) |
| `es_black_friday` | `bool` | `True` si es Black Friday |
| `es_cyber_monday` | `bool` | `True` si es Cyber Monday |
| `es_primer_dia_mes` | `bool` | `True` si es el primer día del mes |
| `es_ultimo_dia_mes` | `bool` | `True` si es el último día del mes |
| `black_friday` | `bool` | Alias de `es_black_friday` usado por el modelo |
| `unidades_vendidas_lag_1` … `lag_7` | `float` | Ventas del día anterior hasta 7 días atrás |
| `unidades_vendidas_ma7` | `float` | Media móvil de las últimas 7 observaciones |
| `descuento_porcentaje` | `float` | Descuento aplicado: `(1 - precio_venta / precio_base) × 100` |
| `precio_competencia` | `float` | Media de los precios de Amazon, Decathlon y Deporvillage |
| `ratio_precio` | `float` | Ratio precio propio / precio de la competencia |
| `nombre_h_*` | `int` (0/1) | One-hot encoding del nombre del producto |
| `categoria_h_*` | `int` (0/1) | One-hot encoding de la categoría |
| `subcategoria_h_*` | `int` (0/1) | One-hot encoding de la subcategoría |

---

## Metodología de Trabajo

El flujo de trabajo sigue una lógica end-to-end:

```
Datos Raw
   │
   ├─ ventas.csv ──┐
   │               ├──► merge (fecha + producto_id) ──► df
   └─ competencia.csv ┘
                         │
                         ▼
               Feature Engineering
               (temporales, lags, calendario,
                precios, one-hot encoding)
                         │
                         ▼
               Split temporal
               ┌──────────────────────────────────┐
               │  Train: 2021–2023 (3 años)       │
               │  Validación: 2024 (1 año)         │
               └──────────────────────────────────┘
                         │
                         ▼
               HistGradientBoostingRegressor
               (learning_rate=0.05, max_iter=400,
                max_depth=6, l2=1.0)
                         │
                         ├─ Evaluar en validación 2024
                         │
                         ▼
               Reentrenamiento final con 2021–2024
                         │
                         ▼
               modelo_Final.joblib
                         │
                         ▼
               Predicción recursiva noviembre 2025
                         │
                         ▼
               App Streamlit (simulación de escenarios)
```

### Pasos detallados

1. **Consolidación de datos**: merge de ventas históricas y precios de competencia por `fecha` y `producto_id`. Sin valores nulos en ninguna de las fuentes.
2. **Análisis exploratorio (EDA)**: distribuciones de ventas, patrones por categoría y subcategoría, estacionalidad semanal, efecto del Black Friday, correlación entre variables de precio.
3. **Preparación de features**: variables de calendario (festivos España via `holidays`), lags (1–7 días), media móvil (7 días), ratio de precio y codificación one-hot de producto, categoría y subcategoría.
4. **Split temporal**: entrenamiento en 2021–2023, validación en 2024. Se evita el data leakage usando exclusivamente el pasado para predecir el futuro.
5. **Entrenamiento y evaluación**: `HistGradientBoostingRegressor` con early stopping. Comparación contra un baseline naïve (media del set de entrenamiento).
6. **Backtesting en noviembre 2024**: predicción de los 7 productos estrella para validar el comportamiento del modelo en el período más relevante (incluye Black Friday).
7. **Reentrenamiento final**: el modelo se vuelve a ajustar con todos los datos disponibles (2021–2024) antes de generar las predicciones de noviembre 2025.
8. **Generación de predicciones recursivas**: para cada día de noviembre 2025, los lags se actualizan con el valor predicho del día anterior, simulando un forecasting real sin ground truth.
9. **Aplicación Streamlit**: interfaz interactiva para simular distintos niveles de descuento y escenarios de precio de la competencia.

---

## Ingeniería de Features

### Variables de calendario

Las variables temporales se calculan a partir de `fecha` y están diseñadas para capturar patrones estacionales relevantes para el comercio minorista deportivo:

- **Efectos de semana**: día de la semana, fin de semana vs. laborable.
- **Efectos de mes**: primer y último día del mes, número de semana, trimestre, semestre.
- **Eventos especiales**: Black Friday (4.º jueves de noviembre), Cyber Monday (lunes siguiente al Black Friday), festivos nacionales de España (librería `holidays`).

### Lags y media móvil

Los lags capturan la autocorrelación de la demanda:

| Feature | Descripción |
| --- | --- |
| `unidades_vendidas_lag_1` | Ventas del día anterior |
| `unidades_vendidas_lag_2` … `lag_7` | Ventas de 2 a 7 días atrás |
| `unidades_vendidas_ma7` | Media de los últimos 7 días |

### Variables de precio

| Feature | Fórmula |
| --- | --- |
| `descuento_porcentaje` | `(1 - precio_venta / precio_base) × 100` |
| `precio_competencia` | Media de `Amazon`, `Decathlon`, `Deporvillage` |
| `ratio_precio` | `precio_venta / precio_competencia` |

Un `ratio_precio` > 1 indica que el precio propio supera al promedio de la competencia.

---

## Modelo Predictivo

### Algoritmo seleccionado

**`HistGradientBoostingRegressor`** de scikit-learn. Se eligió por:

- Soporte nativo para valores faltantes (útil durante la predicción recursiva).
- Buen rendimiento con datos tabulares de tamaño mediano.
- Early stopping integrado que evita el sobreajuste sin búsqueda exhaustiva de hiperparámetros.

### Hiperparámetros del modelo final

| Hiperparámetro | Valor | Justificación |
| --- | --- | --- |
| `learning_rate` | `0.05` | Tasa de aprendizaje conservadora |
| `max_iter` | `400` | Número máximo de estimadores |
| `max_depth` | `6` | Profundidad máxima de cada árbol |
| `l2_regularization` | `1.0` | Regularización para reducir sobreajuste |
| `early_stopping` | `True` | Detención temprana por validación interna |
| `random_state` | `42` | Reproducibilidad |

### Split de entrenamiento

| Set | Periodo | Registros |
| --- | --- | --- |
| Entrenamiento | 2021–2023 | ~2 664 |
| Validación | 2024 | ~888 |
| Modelo final | 2021–2024 | 3 552 |

---

## Métricas de Evaluación

### Validación en 2024 (set de holdout)

| Métrica | Modelo | Baseline Naïve |
| --- | --- | --- |
| **MAE** | **0.72** | 3.34 |
| **RMSE** | **1.51** | 6.25 |
| **MAPE** | **16.44%** | 86.67% |
| **R²** | **0.94** | ~0.00 |

El modelo supera ampliamente al baseline naïve en todas las métricas. Un R² de 0.94 indica que el modelo explica el 94% de la varianza en las unidades vendidas.

### Backtesting — Noviembre 2024 (productos estrella)

Se evaluó el rendimiento del modelo específicamente en el mes de noviembre (mes objetivo) para los 7 productos con mayor volumen de ventas histórico. El análisis se descompuso en tres períodos de 10 días (1–10, 11–20, 21–30) para detectar posible degradación del error en el tiempo.

---

## Predicción Recursiva

Para noviembre 2025 no se dispone de ventas reales, por lo que el modelo genera predicciones **día a día** actualizando los lags con los valores predichos:

```
Día 1 → predicción usando lags históricos reales
Día 2 → predicción usando lag_1 = predicción del día 1
Día 3 → predicción usando lag_1 = predicción del día 2, lag_2 = predicción del día 1
...
Día 30 → todos los lags se nutren de predicciones previas
```

Este enfoque propaga la incertidumbre a lo largo del mes, por lo que las predicciones más alejadas tienen un margen de error potencialmente mayor.

---

## Aplicación Streamlit

### Controles disponibles

| Control | Descripción |
| --- | --- |
| **Producto** | Selector de los 24 productos de la cartera |
| **Ajuste de Descuento** | Slider de −50% a +50% sobre el descuento base (paso: 5 pp) |
| **Escenario de Competencia** | Radio button: Actual, Competencia −5%, Competencia +5% |

### Secciones del dashboard

1. **KPIs resumidos**: unidades totales, ingresos proyectados, precio promedio, descuento promedio.
2. **Gráfico de predicción diaria**: evolución de ventas durante noviembre 2025 con marcador especial para el Black Friday (día 28).
3. **Tabla detallada por día**: precios, descuento, unidades predichas e ingresos, con el Black Friday resaltado.
4. **Comparativa de escenarios de competencia**: comparación automática de los tres escenarios de precio manteniendo el mismo descuento seleccionado.

### Lógica de simulación

Cuando el usuario modifica el descuento:

```
precio_venta = precio_base × (1 − descuento_porcentaje / 100)
ratio_precio = precio_venta / precio_competencia
```

Cuando modifica el escenario de competencia se aplica un factor multiplicativo:

| Escenario | Factor |
| --- | --- |
| Actual (0%) | ×1.00 |
| Competencia −5% | ×0.95 |
| Competencia +5% | ×1.05 |

---

## Cómo Reproducir el Proyecto

### Requisitos previos

- Python 3.9+
- Git

### Pasos

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

4. *(Opcional)* Reentrenar el modelo ejecutando los notebooks en orden:
   - `notebooks/entrenamiento.ipynb`
   - `notebooks/forecasting.ipynb`

5. Ejecutar la aplicación:

   ```powershell
   streamlit run app/app.py
   ```

### Dependencias principales (`requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
holidays
```

---

## Principales Hallazgos

- Se construyó un flujo de predicción diaria orientado a negocio, con foco en **24 productos** y escenarios comerciales concretos para noviembre 2025.
- El modelo `HistGradientBoostingRegressor` alcanzó un **R² de 0.94** en el set de validación, reduciendo el MAE de 3.34 (naïve) a **0.72 unidades**.
- El proyecto integra variables de calendario relevantes para retail, como el efecto del **Black Friday** (día 28 de noviembre), que produce picos de demanda significativos en los productos estrella.
- La predicción recursiva permite generar un forecast completo de 30 días actualizando los lags automáticamente.
- La simulación permite comparar rápidamente el impacto de cambios de descuento y de competencia sobre **unidades e ingresos proyectados**.
- El dashboard entrega una salida accionable mediante **KPIs, tabla diaria y comparativa de escenarios**, interpretable por usuarios no técnicos.

---

## Valor para una Empresa

- **Planificación comercial**: estimación anticipada de la demanda para ajustar el stock antes de un mes crítico como noviembre.
- **Decisiones de pricing basadas en datos**: comparar escenarios de descuento y competencia cuantitativamente, reduciendo la dependencia de la intuición.
- **Puente entre negocio y datos**: la app facilita conversaciones entre equipos de operaciones, comercial y análisis a partir de un mismo modelo compartido.
- **Herramienta accesible**: convierte un modelo predictivo complejo en una interfaz utilizable por usuarios no técnicos sin necesidad de conocimientos de programación.
