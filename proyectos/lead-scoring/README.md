# Lead Scoring | Priorizar contactos con evaluación explícita

[← Volver al portfolio](../../README.md) · Machine Learning · Informe de lectura en línea

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
| Interfaz de inferencia | FastAPI local y procesamiento por lotes | Implementación técnica; no servicio público desplegado. |

## Cómo se abordó

1. Entrenar con validación cruzada estratificada y un pipeline que incluye el escalado dentro de los pliegues.
2. Comprobar identidad de los datos y separación de identificadores entre entrenamiento y validación.
3. Registrar rendimiento y cobertura en la reserva y preparar inferencia por lotes y una API local con validación de entradas.

Los resultados de selección de variables e hiperparámetros por validación cruzada son exploratorios. La cifra destacada de esta página procede del informe reservado, no de las curvas exploratorias.

## Decisión que informa

- Investigar si el ordenamiento puede ayudar a distribuir una capacidad comercial limitada.
- Definir un umbral de acción según capacidad, costes y consecuencias de falsos positivos y falsos negativos.
- Verificar disponibilidad temporal de las variables y evaluar una prueba operativa antes de usar el modelo con contactos reales.

## Límites que cambian la interpretación

- Campos de actividad, como SMS enviados o correos abiertos, requieren verificar su disponibilidad al momento de puntuar; no se presupone que sean válidos para todos los usos.
- La métrica no cubre los 48 contactos no evaluados y no prueba beneficio económico, calibración ni estabilidad futura.
- Los valores son resultados guardados en el proyecto; no se volvieron a ejecutar para preparar esta página.
- No se anuncia una API pública ni experiencia operativa en producción. No se publican contactos, identificadores ni datos personales.

## Qué demuestra el proyecto

Pipeline de modelado, separación de conjuntos, cobertura de evaluación y preparación de inferencia. La principal conclusión de despliegue es que la disponibilidad temporal de los datos debe resolverse antes del uso operativo.

**Estado de esta entrega:** conclusiones disponibles sin descargar nada. La API permanece local; esta página no simula predicciones ni ofrece una demo ficticia.
