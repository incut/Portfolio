# Mercado inmobiliario | Un filtro exploratorio para CABA

[← Volver al portfolio](../../README.md) · Business Analytics · Informe de lectura en línea

**[Abrir el informe HTML original completo y sus mapas](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/mercado-inmobiliario/informe-completo.html)** · [Análisis completo: 19 secciones y 36 figuras](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/mercado-inmobiliario/analisis-completo.html)

Los enlaces anteriores permiten leer o explorar el material completo en el navegador, sin descargar archivos. Esta ficha es solo una síntesis.

**Conclusión:** comparar barrios no alcanza: conviene contrastar ocupación estimada, características del inmueble y costo de compra. El análisis construye un filtro de investigación, no una lista de inversiones recomendadas.

## La pregunta de negocio

¿Cómo integrar información de alquiler temporario y precios inmobiliarios para identificar perfiles que merezcan una evaluación más detallada?

## Resultados del análisis

| Indicador | Resultado | Interpretación |
| --- | --- | --- |
| Anuncios analizados | 12.974 | Universo observado, no censo completo del mercado. |
| Anuncios con rendimiento estimado | 12.435 | Subconjunto con información suficiente para el cálculo. |
| Rendimiento bruto mediano estimado | 5,59% | Estimación bajo los supuestos del análisis, no rentabilidad neta. |
| Mediana estimada de San Nicolás | 9,85% | Comparación descriptiva del barrio dentro de la muestra. |
| Anuncios que cumplen el filtro | 2.467 | Candidatos de análisis, no oportunidades verificadas de inversión. |

El filtro combina rendimiento bruto estimado ≥11,61% y ocupación estimada ≥160 noches. Son umbrales del estudio, no condiciones suficientes para comprar. El informe original denomina «margen» al indicador de rendimiento; no debe confundirse con margen neto.

## Cómo se abordó

1. Integrar anuncios de Inside Airbnb Buenos Aires y fuentes de precios y geografía.
2. Construir indicadores comparables por barrio y características del inmueble.
3. Analizar perfiles y aplicar umbrales para acotar una investigación posterior.

Las fuentes documentadas incluyen Inside Airbnb del 29/06/2026, ARED de enero–febrero de 2024, precios por m² de IDECBA/GCBA del primer trimestre de 2026 y límites oficiales de barrios. La combinación de periodos requiere cautela.

## Decisión que informa

- Comparar perfiles concretos, no seleccionar únicamente por la popularidad del barrio.
- Someter el resultado a escenarios de ocupación, precio de compra y tipo de cambio.
- Verificar cada inmueble, sus gastos y las condiciones aplicables antes de cualquier decisión real.

## Límites que cambian la interpretación

- La ocupación es estimada; no corresponde a reservas observadas directamente y forma parte del cálculo del ingreso.
- El rendimiento bruto excluye gastos, mantenimiento, impuestos, comisiones y financiación.
- Las fuentes tienen fechas distintas; las cifras no describen automáticamente el mercado actual.
- La asociación entre ocupación y rendimiento no prueba causalidad: ambos indicadores comparten componentes del cálculo.

## Qué demuestra el proyecto

Integración de fuentes, análisis geográfico y explicitación de supuestos. Se publican únicamente cifras agregadas; no microdatos, direcciones ni mapas de anuncios individuales. Esta síntesis no constituye una tasación ni asesoramiento de inversión.

**Estado de esta entrega:** informe legible en el navegador; resultados históricos del estudio, no un buscador de oportunidades actualizado.

## Lectura y estructura

Esta carpeta contiene la ficha `README.md` con el objetivo, el enfoque, los resultados y sus límites. Los enlaces superiores abren el resumen y los materiales completos en el navegador; no requieren instalación. No se redistribuyen aquí notebooks, datasets ni modelos, por lo que esta ficha no permite reproducir el entrenamiento o los cálculos originales.

## Barrios y fuentes

San Nicolás (9,85%), Monserrat (7,43%) y San Telmo (7,08%) lideran por rendimiento bruto mediano estimado. Los 2.467 candidatos representan aproximadamente el 19,0% de todos los 12.974 anuncios analizados, no de las transacciones del mercado.

La documentación identifica Inside Airbnb, corte 29/06/2026, y referencias agregadas de precio por m² de IDECBA. Los costos de adquisición son aproximaciones; no se atribuyen estos resultados a los microdatos complementarios de venta ARED. [Consultar fuentes y limitaciones](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/mercado-inmobiliario/#evidencia).
