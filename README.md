# Augusto Presfraind | Analítica de datos y machine learning

[LinkedIn](https://ar.linkedin.com/in/augusto-presfraind) · [GitHub](https://github.com/incut) · [Correo](mailto:augustopresfraind@gmail.com)

## De los datos a una decisión de negocio

Busco oportunidades junior en ciencia de datos y analítica de negocio. Estos proyectos muestran cómo preparo datos, evalúo modelos y traduzco hallazgos en decisiones con limitaciones explícitas.

**No es necesario descargar archivos ni instalar herramientas.** Las fichas de GitHub ofrecen una entrada breve. Las presentaciones, informes e insights completos se leen en el sitio público, con sus gráficos y limitaciones originales.

**[Abrir el portfolio y explorar los proyectos](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/)**

## Casos finalizados

| Proyecto | Pregunta de negocio | Resultado destacado | Qué demuestra | Acceso |
| --- | --- | --- | --- | --- |
| **Lead Scoring** · Machine Learning | ¿Cómo ordenar contactos comerciales según su propensión a convertir? | ROC-AUC registrada de 0,8969 en 2.680 contactos evaluables de la reserva de validación. | Selección de variables, validación y API de inferencia | [Caso breve](proyectos/lead-scoring/README.md) · [Material completo](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/lead-scoring/) |
| **Plantas solares** · Business Analytics | ¿Dónde priorizar la investigación de pérdidas potenciales de generación? | La planta P2 concentra el 98,4% de la energía potencialmente no generada estimada en la muestra. | Calidad de datos, anomalías y comunicación de escenarios | [Caso breve](proyectos/plantas-solares/README.md) · [Material completo](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/plantas-solares/) |
| **Ecommerce** · Business Analytics | ¿Cómo priorizar la retención y la segunda compra? | El 21,22% de compradores recurrentes aporta el 46,72% de los ingresos registrados. | Segmentación RFM, cohortes y priorización comercial | [Caso breve](proyectos/ecommerce/README.md) · [Material completo](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/ecommerce/) |
| **Mercado inmobiliario** · Business Analytics | ¿Cómo construir un filtro exploratorio de alquiler temporario en CABA? | 12.974 anuncios analizados, con supuestos explícitos de ocupación y rendimiento bruto. | Integración de fuentes, análisis geográfico y supuestos | [Caso breve](proyectos/mercado-inmobiliario/README.md) · [Material completo](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/mercado-inmobiliario/) |

> Las cifras pertenecen a cada análisis; no representan resultados comerciales obtenidos en producción. Los escenarios, estimaciones y métricas de validación se identifican dentro de cada caso.

## Recorrido sugerido

- **Negocio y comunicación:** comenzar por [Ecommerce](proyectos/ecommerce/README.md) y continuar con [Plantas solares](proyectos/plantas-solares/README.md).
- **Modelado y evaluación:** revisar [Lead Scoring](proyectos/lead-scoring/README.md), incluyendo cobertura y restricciones de uso.
- **Integración de fuentes y supuestos:** explorar [Mercado inmobiliario](proyectos/mercado-inmobiliario/README.md).

## Herramientas utilizadas en los proyectos

| Área | Herramientas |
| --- | --- |
| Preparación y análisis | Python, pandas, NumPy, SQL |
| Modelado | scikit-learn |
| Visualización y comunicación | Matplotlib, Seaborn, informes ejecutivos |
| Prototipos de aplicación | Streamlit, FastAPI |
| Trabajo reproducible | Jupyter, Git, GitHub |

## Qué se publica

Cada proyecto tiene un resumen breve y acceso a los informes, gráficos y presentaciones completos. Se publican resultados agregados, sin registros de clientes, leads reales ni credenciales. Los resultados estimados y las limitaciones se distinguen de los efectos medidos en producción.

## Lectura, estructura y reproducción

Los cuatro casos son documentación: se exploran desde el navegador, sin instalar Python. `proyectos/<caso>/README.md` contiene el objetivo y los resultados; `assets/` contiene las figuras de síntesis. El sitio enlazado ofrece las presentaciones, análisis y mapas completos.

**Lead Scoring incluye ahora una [demostración técnica ejecutable](proyectos/lead-scoring/technical/README.md)** con datos sintéticos, pipelines, validación cruzada anidada y pruebas. La [comparación sobre entrenamiento real](proyectos/lead-scoring/training_comparison.md) se publica por separado como evidencia agregada, no como resultado de la demostración sintética.

Los datasets reales, modelos privados y notebooks originales de los cuatro casos no se redistribuyen. Los otros tres casos siguen siendo publicaciones de lectura, no entornos reproducibles de cálculo. La demostración de Lead Scoring tampoco permite reconstruir las métricas reales sin los datos privados.

La aplicación educativa independiente [ForecastingVentas](ForecastingVentas/README.md) conserva su propia documentación y su [archivo de dependencias](ForecastingVentas/requirements.txt). No comparte automáticamente el entorno de los cuatro casos.

## Proyecto previo y trabajo en curso

- [ForecastingVentas: documentación](ForecastingVentas/README.md) y [demo educativa existente](https://portfolio-forecasting-ventas.streamlit.app). **En revisión metodológica:** sus métricas históricas no deben interpretarse como evidencia de capacidad predictiva validada. La demo es un prototipo, no una recomendación comercial.
- **Forecasting_Retail:** en desarrollo; no se incluye entre los casos finalizados.

Las cuatro páginas principales ofrecen informes de lectura en línea. **Lead Scoring también tiene una API desplegada en Render:** [probar en Swagger](https://lead-scoring-anta.onrender.com/docs). Se incluye un ejemplo sintético en el caso, sin descargar nada.

---

[Perfil de GitHub](https://github.com/incut)

