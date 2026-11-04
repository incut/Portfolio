# Ecommerce | Convertir la primera compra en una relación

[← Volver al portfolio](../../README.md) · Business Analytics · Informe de lectura en línea

**[Ver todos los materiales completos](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/ecommerce/)** · [Presentación original completa](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/ecommerce/informe-ejecutivo.html) · [Todos los insights](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/ecommerce/insights-completos.html) · [Dashboard interactivo](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/ecommerce/dashboard.html)

Los enlaces anteriores permiten leer o explorar el material completo en el navegador, sin descargar archivos. Esta ficha es solo una síntesis.

**Conclusión:** la recurrencia merece atención: el 21,22% de compradores que repite aporta el 46,72% de los ingresos registrados. El siguiente paso propuesto es probar acciones de segunda compra y recuperación, no asumir que una campaña ya generó resultados.

![Los compradores recurrentes representan el 21,22% de compradores y el 46,72% de ingresos; los de una compra representan el 78,78% y el 53,28%, respectivamente.](../../assets/ecommerce.svg)

## La pregunta de negocio

¿Qué segmentos conviene priorizar para retención y recuperación, utilizando comportamiento observado en lugar de una campaña indiferenciada?

## Resultados del análisis

| Indicador | Resultado | Lectura de negocio |
| --- | --- | --- |
| Compradores | 11.040 | Base observada del análisis. |
| Compras aproximadas | 15.452 | Sesiones únicas con evento de compra; no pedidos confirmados. |
| Ingresos registrados | 621.549,60 € | No equivalen a beneficio ni margen. |
| Compradores de una sola compra | 78,78% | Segmento relevante para una hipótesis de segunda compra. |
| Clientes de alto gasto y menor recencia | 1.565 | Grupo identificado para evaluar acciones de recuperación. |

## Cómo se abordó

1. Integrar y revisar eventos de cinco meses, de octubre de 2019 a febrero de 2020.
2. Construir una aproximación de compra por sesión y analizar recurrencia y cohortes.
3. Segmentar con RFM: recencia, frecuencia y valor monetario; vincular cada grupo con una acción posible.

## Decisión que informa

- Diferenciar mensajes de segunda compra, fidelización y recuperación según el comportamiento del segmento.
- Diseñar una prueba con grupo de comparación para medir el efecto incremental de una acción.
- Incorporar margen, costo de contacto y devoluciones antes de decidir presupuesto.

## Límites que cambian la interpretación

- La sesión con evento de compra es una aproximación; no sustituye a un identificador de pedido.
- Cinco meses no permiten afirmar retención de largo plazo ni comportamiento anual.
- Los escenarios de segunda compra representan ingresos brutos potenciales, no aumento causal demostrado.
- No se dispone de costos, márgenes ni devoluciones suficientes para demostrar rentabilidad de una campaña.

## Qué demuestra el proyecto

Análisis de clientes, segmentación RFM, cohortes y traducción de hallazgos a hipótesis de negocio comprobables. Esta síntesis no publica registros de clientes ni ofrece resultados de campañas ejecutadas.

**Estado de esta entrega:** conclusiones y visualización agregada disponibles en el navegador, sin instalación ni descarga.

## Lectura y estructura

Esta carpeta contiene la ficha `README.md` con el objetivo, el enfoque, los resultados y sus límites. Los enlaces superiores abren el resumen y los materiales completos en el navegador; no requieren instalación. No se redistribuyen aquí notebooks, datasets ni modelos, por lo que esta ficha no permite reproducir el entrenamiento o los cálculos originales.

## Dimensionar la segunda compra

Un aumento ilustrativo de un punto porcentual de compradores recurrentes sobre 11.040 compradores representa unas 110 personas. Suponiendo un pedido adicional por persona y un ticket aproximado de 34,32 €, serían unos 3.789 € brutos, no beneficio ni impacto causal medido.

El recomendador compara los 100 productos más comprados por distancia euclidiana y devuelve cinco sugerencias por defecto. No tiene métricas de ranking ni mejora comercial evaluadas. [Ver los escenarios originales y sus supuestos](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/ecommerce/#evidencia).
