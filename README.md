# Siniestros viales en CABA: predicción de la gravedad

Trabajos del curso de Data Science. El proyecto principal clasifica cada siniestro vial registrado en la Ciudad de Buenos Aires como **LEVE**, **GRAVE** o **MORTAL**. Para eso usa datos de cuándo ocurrió, dónde, quiénes fueron las víctimas y contra qué chocaron.

**Fuente:** [Buenos Aires Data, Víctimas de siniestros viales](https://data.buenosaires.gob.ar/dataset/victimas-siniestros-viales). Se usan las tablas `hechos` (un siniestro por fila) y `victimas` (una persona por fila), relacionadas uno a muchos por `id_siniestro`.

## Contenido

| Archivo | Qué contiene |
|---|---|
| `01_eda_y_limpieza_profunda.ipynb` | Diagnóstico de calidad: estructura, faltantes (incluidos los disfrazados, como `"SD"` y `"#¡REF!"`), duplicados, categorías mal escritas, edad, atípicos y coherencia interna. Después aplica la limpieza que ese diagnóstico justifica. |
| `02_modelo_clasificacion_gravedad.ipynb` | **3ª pre-entrega.** Cubre todo el proceso: limpieza, EDA orientado al objetivo, preparación de variables, Regresión Logística y Random Forest, optimización con `GridSearchCV`, evaluación y selección del modelo. |
| `Informe EDA/` | Informe del EDA en Word (versión completa y resumen) y sus gráficos. |
| `Guía para limpieza y normalización de datos.ipynb` | Guía de referencia del curso. |
| `Notebook_18_C14.ipynb`, `Santiago_Ruiz_practica_numpy5.ipynb` | Prácticas de clase. |

Los CSV (crudos y limpios) no se incluyen en el repositorio. Se descargan desde la fuente y los notebooks generan `hechos_limpio.csv` y `victimas_limpio.csv`.

## Decisiones principales de limpieza

- **El crudo no se modifica.** Todo se hace sobre copias, y cada paso imprime cuántos valores cambió.
- **Normalización de categorías.** Primero tipográfica (espacios, mayúsculas, acentos) y después semántica con diccionarios de equivalencias (`OMNIBUS → COLECTIVO`, `MOTOCICLETA → MOTO`).
- **Valores imposibles a faltante**, sin borrar filas: edad mayor a 110, hora fuera de 0–23, comuna fuera de 1–15.
- **Los faltantes se tratan según lo que significa cada variable.** Las categóricas reciben el nivel `SIN DATOS`. Los conteos vacíos pasan a 0. La edad se imputa con la mediana y queda marcada. La fecha de fallecimiento, las coordenadas y la comuna no se imputan.
- **Los duplicados exactos de `victimas` se conservan**, porque son personas distintas con los mismos atributos.
- **La variable objetivo se deriva de los conteos de víctimas** (MORTAL > GRAVE > LEVE), que se pueden verificar aritméticamente. No se usa la etiqueta de texto.

## Modelo

- **Unidad:** el siniestro (65.818 casos). Se excluyen las variables que filtran la respuesta (*data leakage*): conteos por gravedad, gravedad de la víctima y fecha de fallecimiento.
- **Predictoras:** hora, día de la semana, mes, año, cantidad de víctimas, edad media, tipo de vía, modo de desplazamiento, contraparte y comuna (con one-hot).
- **Desbalance:** alrededor de 94 % LEVE, 4,5 % GRAVE y 1 % MORTAL. Por eso se usa una división train/test 80/20 estratificada, `class_weight` y **F1 macro** como métrica principal.
- **Optimización:** `GridSearchCV` con validación cruzada estratificada de 3 pliegues.

### Resultados en test

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---|---|---|
| Línea de base (siempre LEVE) | 0,944 | 0,315 | 0,333 | ≈ 0,32 |
| Regresión Logística (optimizada) | 0,641 | 0,374 | 0,606 | 0,343 |
| Random Forest (sin restricciones) | 0,944 | 0,497 | 0,346 | — (sobreajusta) |
| **Random Forest (optimizado)** | 0,899 | 0,457 | 0,542 | **≈ 0,48** |

**Modelo elegido: Random Forest optimizado** (`min_samples_leaf=5`, `class_weight="balanced_subsample"`). Tiene el mejor F1 macro en validación cruzada (0,484) y en test. La Regresión Logística detecta más casos graves y mortales (recall 0,46 y 0,71), pero da demasiadas falsas alarmas (precision de 0,10 y 0,05). El Random Forest equilibra mejor la detección y la precisión.

### Hallazgos

- La proporción de siniestros graves o mortales **se triplica de madrugada**: alrededor de 16 % a las 3 h, contra 4 % a la tarde. También es mayor los fines de semana y en autopistas.
- El **peatón** es la víctima más expuesta, seguido por la moto y el monopatín. Chocar contra un **objeto fijo** es lo más grave, y el **camión** concentra la mayor proporción de siniestros mortales.
- La categoría `SIN DATOS` aparece casi siempre en siniestros leves, porque esos casos se registran con menos detalle. Es un sesgo de la fuente que el modelo también aprende.

## Herramientas

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter / Google Colab

## Autor

Santiago Ruiz · [GitHub](https://github.com/santinicolas1998) · [LinkedIn](https://linkedin.com/in/santiago-ruiz-180373127)
