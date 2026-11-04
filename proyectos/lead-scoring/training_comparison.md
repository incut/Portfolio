# Comparación de entrenamiento, separada de la demostración sintética

Ejecutada localmente el 4 de noviembre de 2026 sobre los mismos **6.365 leads de
entrenamiento histórico**, con 2.393 compradores. No se evaluó ni se utilizó la
reserva para seleccionar modelos. Los originales y la separación se conservaron.

| Métrica OOF agregada | Formulario | Ampliado no verificado temporalmente |
| --- | ---: | ---: |
| ROC-AUC | 0,761223 | 0,899605 |
| Lift@10 % | 2,467765 | 2,542925 |
| Precisión del top 10 % | 0,927786 | 0,956044 |
| Compradores capturados | 591 de 2.393 | 609 de 2.393 |
| Leads priorizados | 637 | 637 |
| Cobertura | 100 % | 100 % |

Ambos pipelines nuevos usaron regresión logística L1, cinco folds externos
estratificados iguales y tres internos para elegir C entre 0,1, 1 y 10 por AUC.
Imputación, codificación y escalado se ajustaron dentro de cada fit, sin RFECV
global ni filtro que exija actividad. El ampliado usa diez campos crudos, no es
una reproducción del modelo histórico desplegado de 28 variables seleccionadas.

Lift = precisión en los primeros techo(0,10*n) / prevalencia de la misma población.
Los empates se resuelven por orden fuente, independientemente del target. Las
predicciones OOF agregadas proceden de varios modelos, no de una prueba externa.
El entrenamiento ya había informado exploraciones previas, esta CV no borra ese uso.

Estos resultados **no proceden de datos sintéticos**, no prueban impacto causal
ni disponibilidad de SMS o actividad al completar formulario. La AUC reservada
histórica 0,8969 y su cobertura 98,24 % siguen siendo otra evidencia, no una
comparación directa por diferencias de población y construcción del modelo.

La [demostración pública](technical/README.md) permite inspeccionar y ejecutar
el esquema de evaluación, no reconstruir estas cifras sin los datos privados.
Pendiente: evidencia del sistema origen sobre captura/actualización de campos y
madurez del resultado. Una prueba final genuinamente nueva requiere futuros leads.
