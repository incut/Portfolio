# Lead Scoring | Priorizar contactos con evaluación explícita

[← Volver al portfolio](../../README.md) · Machine Learning · Informe de lectura en línea

**[Ver resumen y análisis completo](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/lead-scoring/)** · [Informes originales](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/lead-scoring/informes.html) · [Todos los análisis y figuras](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/lead-scoring/anexos.html) · [Probar la API en Render](https://lead-scoring-anta.onrender.com/docs)

Los enlaces anteriores permiten leer o explorar el material completo en el navegador, sin descargar archivos. Esta ficha es solo una síntesis.

**Resultado:** el informe de validación reservado registra una ROC-AUC de **0,8969**, calculada sobre **2.680 de 2.728 contactos**. Es una medida de discriminación en los casos evaluables, no precisión del 89,69% ni mejora comercial demostrada.

> **Restricción de uso:** antes de puntuar un contacto en el momento de completar un formulario, falta verificar que todas las variables utilizadas ya estén disponibles en ese instante. El resultado no basta para afirmar que el modelo esté listo para producción.

## La pregunta de negocio

¿Cómo ordenar contactos comerciales por propensión a convertir y evaluar el sistema sin confundir una buena métrica con resultados de ventas?

## Resultados documentados

| Indicador | Resultado | Interpretación |
| --- | --- | --- |
| ROC-AUC de la reserva | 0,8969 | Discriminación registrada sobre el subconjunto evaluable. |
| Contactos reservados | 2.728 | Conjunto destinado a validación. |
| Contactos evaluados | 2.680 | 48 contactos no forman parte de la métrica publicada. |
| Cobertura de evaluación | 98,24% | Debe leerse junto con la ROC-AUC, no omitirse. |
| Interfaz de inferencia | FastAPI desplegada en Render y procesamiento por lotes | Prototipo accesible; no demuestra por sí solo experiencia operativa ni impacto comercial. |

## Cómo se abordó

1. Entrenar con validación cruzada estratificada y un pipeline que incluye el escalado dentro de los pliegues.
2. Comprobar identidad de los datos y separación de identificadores entre entrenamiento y validación.
3. Registrar rendimiento y cobertura en la reserva y preparar inferencia por lotes y una API con validación de entradas, cuyo prototipo está desplegado en Render.

Los resultados de selección de variables e hiperparámetros por validación cruzada son exploratorios. La cifra destacada de esta página procede del informe reservado, no de las curvas exploratorias.

## Decisión que informa

- Investigar si el ordenamiento puede ayudar a distribuir una capacidad comercial limitada.
- Definir un umbral de acción según capacidad, costos y consecuencias de falsos positivos y falsos negativos.
- Verificar disponibilidad temporal de las variables y evaluar una prueba operativa antes de usar el modelo con contactos reales.

## Límites que cambian la interpretación

- Campos de actividad, como SMS enviados o correos abiertos, requieren verificar su disponibilidad al momento de puntuar; no se presupone que sean válidos para todos los usos.
- La métrica no cubre los 48 contactos no evaluados y no prueba beneficio económico, calibración ni estabilidad futura.
- Los valores son resultados guardados en el proyecto; no se volvieron a ejecutar para preparar esta página.
- La API está públicamente accesible como prototipo. No se atribuye por ello experiencia operativa en producción ni beneficio comercial. No se publican contactos reales ni datos personales.

## Qué demuestra el proyecto

Pipeline de modelado, separación de conjuntos, cobertura de evaluación y preparación de inferencia. La principal conclusión de despliegue es que la disponibilidad temporal de los datos debe resolverse antes del uso operativo.

**Estado de esta entrega:** conclusiones y API pública accesibles desde el navegador. La comprobación del 2 de noviembre de 2026 obtuvo HTTP 200 en /health y en una única predicción sintética. La disponibilidad puede variar y el servicio puede necesitar tiempo para iniciarse.


## Probar la API sin descargar nada

1. Abrir [Swagger en Render](https://lead-scoring-anta.onrender.com/docs).
2. Expandir `POST /predict` y seleccionar **Try it out**.
3. Pegar el ejemplo sintético siguiente y seleccionar **Execute**. No ingresar datos personales.

```json
[
  {
    "id": "demo-lead-001",
    "no_enviar_email": "No",
    "no_llamar": "No",
    "origen": "API",
    "fuente": "Chat",
    "ult_actividad": "Page Visited on Website",
    "ambito": "Select",
    "ocupacion": "Unemployed",
    "visitas_total": 0.0,
    "tiempo_en_site_total": 0,
    "paginas_vistas_visita": 0.0,
    "score_actividad": 15.0,
    "score_perfil": 15.0
  }
]
```

El ejemplo usa un identificador ficticio. La respuesta incluye la probabilidad, la clasificación y el canal asignado. [Consultar el estado del servicio](https://lead-scoring-anta.onrender.com/health).

## Lectura y estructura

Esta ficha conserva el objetivo, los resultados y sus límites. La [entrega técnica](technical/README.md) añade código ejecutable, generación sintética y pruebas. La [comparación nueva sobre entrenamiento](training_comparison.md) y el [diccionario de disponibilidad](availability.md) separan evidencia real de demostración. No se redistribuyen datos reales, modelos privados ni notebooks originales, y la API desplegada no se modifica.

## Qué significan las métricas y los canales

La AUC reservada de 0,8969 corresponde a 2.680 de 2.728 contactos. El lift numérico del decil superior reservado no está disponible en la evidencia revisada.

**Riesgo temporal:** `SMS enviado` es la tercera variable por importancia de permutación. Si ocurre después del momento de puntuación, no puede usarse para predecir antes del contacto. Ese uso no está validado; tampoco se documentó una ablación que quite únicamente SMS.

La comparación exploratoria disponible usa la misma regresión logística: 21 variables de formulario, AUC CV 0,757242; escenario ampliado de 28 variables, 0,896819. No es una comparación independiente de algoritmos en la reserva. Los canales usan reglas de negocio: p < 0,4 autogestión; 0,4 ≤ p ≤ 0,7 automatización; p > 0,7 comercial. No se optimizaron por costos.

[Ver métricas, reglas y demostración con un contacto ficticio](https://incut-portfolio-completo.augustopresfraind.chatgpt.site/lead-scoring/#evidencia). La etapa 06 se omitió porque no se aplicó balanceo de clases.

