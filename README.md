# Sistema inteligente de valoración de mercado, detección de fichajes prioritarios y perfilamiento táctico en el fútbol europeo

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Desplegado-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?logo=pytorch&logoColor=white)
![Licencia](https://img.shields.io/badge/Licencia%20Datos-CC%20BY--NC--SA%204.0-1B4965)

Proyecto integrador de la asignatura **Ciencia de Datos** — Maestría en Inteligencia Artificial, Facultad de Ingeniería, **Universidad de La Salle** (Bogotá D. C., Colombia).

---

## 1. ¿De qué trata este proyecto?

En el fútbol profesional moderno, fichar o renovar a un jugador implica inversiones de millones de euros que muchas veces se toman por intuición o por el impacto mediático del momento. Esto genera dos riesgos frecuentes para los clubes: pagar sobreprecios o dejar pasar jóvenes talentos antes de que su valor se dispare.

Este proyecto aplica el ciclo completo de la ciencia de datos sobre **1 200 futbolistas de las cinco grandes ligas europeas** (Premier League, LaLiga, Serie A, Bundesliga y Ligue 1) para responder tres preguntas clave de forma objetiva y en segundos:

1. **¿Cuánto vale realmente un futbolista?** (Regresión: estimación del valor de mercado en millones de euros).
2. **¿Es una oportunidad prioritaria de fichaje?** (Clasificación: detección de jugadores de alto impacto).
3. **¿Qué perfil táctico cumple en la cancha?** (Segmentación no supervisada: agrupación por estilo de juego para encontrar reemplazos equivalentes a menor costo).

---

## 2. Aplicativo interactivo en vivo

La solución está desplegada en una aplicación web interactiva construida con **Streamlit**, pensada para que cualquier director deportivo o usuario no técnico pueda explorar los datos sin escribir código:

* **Enlace al tablero en vivo:** [Abrir aplicativo en Streamlit](https://proyecto-ciencia-datos-futbol-fyj9adjhkxxd2s7lap52dg.streamlit.app/)

### Módulos disponibles en el tablero
1. **Filtros dinámicos:** segmentación en tiempo real por liga europea, posición táctica, rango de edad y valoración general mínima.
2. **Tarjetas KPI ejecutivas:** indicadores resumidos que se actualizan al instante según los filtros seleccionados.
3. **Simulador de predicción en vivo:** controles interactivos para ingresar el perfil de cualquier futbolista y calcular al instante su valor justo de mercado y su probabilidad de fichaje prioritario.
4. **Segmentación táctica no supervisada:** mapa espacial de clústeres (PCA 2D), gráfico de radar multidimensional y tabla numérica promedio por arquetipo táctico.

---

## 3. Conjunto de datos

* **Fuente:** Kaggle Datasets (*EA Sports FC / Transfermarkt Football Players Dataset*).
* **Licencia:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
* **Tamaño:** 1 200 registros y 16 variables mixtas (cualitativas nominales y ordinales; cuantitativas discretas y continuas).
* **Calidad y preparación:** cero duplicados, imputación multivariada bajo mecanismo MAR con `KNNImputer`, recorte suave de valores extremos (*winsorization* al 1 % y 99 %) y partición estratificada **70 % entrenamiento / 15 % validación / 15 % prueba** libre de fuga de datos (*data leakage*).
* **Variables creadas a partir del dominio deportivo:**
  * `goal_contributions_per_90`: goles más asistencias por cada 90 minutos jugados.
  * `development_margin`: diferencia entre el potencial proyectado y la valoración actual.
  * `tactical_efficiency_index`: índice combinado de precisión de pase (50 %) y duelos defensivos ganados (50 %).

---

## 4. Resumen de resultados

Todos los modelos fueron evaluados sobre el conjunto de prueba independiente (180 futbolistas no vistos durante el entrenamiento) tras optimizar hiperparámetros con validación cruzada de 5 pliegues.

### Predicción del valor de mercado (regresión)

| Modelo | $R^2$ | RMSE (M€) | MAE (M€) |
| :--- | :---: | :---: | :---: |
| Baseline: regresión lineal | 0,6502 | 6,3513 | 4,1364 |
| Regresión regularizada (*Ridge*) | 0,6533 | 6,3231 | 4,0362 |
| *Random Forest Regressor* | 0,9387 | 2,6587 | 1,7499 |
| Perceptrón multicapa (MLP en PyTorch) | 0,9544 | 2,2920 | 1,5487 |
| ***Gradient Boosting Regressor* (ganador)** | **0,9663** | **1,9724** | **1,3731** |

* **Interpretabilidad (SHAP):** los tres factores con mayor peso en el precio estimado son el potencial proyectado (`potential_rating`), la valoración actual (`overall_rating`) y la edad (`age`), seguidos por la reputación internacional, la liga y las variables creadas.

### Detección de fichajes prioritarios (clasificación)

| Modelo | Exactitud | Precisión | *Recall* | *F1-Score* | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline: regresión logística | 0,9056 | 0,5405 | 1,0000 | 0,7018 | 0,9828 |
| *Gradient Boosting Classifier* | 0,9889 | 0,9091 | 1,0000 | 0,9524 | 1,0000 |
| ***Random Forest Classifier* (ganador)** | **0,9944** | **0,9524** | **1,0000** | **0,9756** | **0,9994** |

### Segmentación táctica no supervisada (*clustering*)

Mediante el Método del Codo y el Coeficiente de Silueta se seleccionó **$K = 4$** con **K-Means** (silueta $= 0,1900$, superior al *clustering* jerárquico aglomerativo con $0,1681$), descubriendo cuatro arquetipos tácticos:

| Perfil táctico | Jugadores | Valoración | Goles | Pase (%) | Velocidad (km/h) | Duelos def. (%) | Valor prom. (M€) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clúster 0:** defensores y volantes de contención | 520 | 68,97 | 0,77 | 84,48 | 30,57 | 62,75 | 4,26 |
| **Clúster 1:** atacantes veloces y finalizadores | 243 | 75,08 | 4,76 | 80,57 | 33,03 | 44,37 | 11,79 |
| **Clúster 2:** porteros y especialistas de cierre | 144 | 72,74 | 0,07 | 76,30 | 25,50 | 74,67 | 8,46 |
| **Clúster 3:** creadores de juego y figuras de élite | 293 | 77,95 | 1,71 | 88,63 | 31,31 | 65,88 | 15,11 |

---

## 5. Estructura del repositorio

```text
proyecto-ciencia-datos-futbol/
├── data/
│   └── football_players_kaggle.csv                  # Conjunto de datos de 1 200 futbolistas
├── notebooks/
│   └── Proyecto_Integrador_Ciencia_Datos_Futbol.ipynb  # Cuaderno ejecutable (Fases 1, 2 y 3)
├── dashboard/
│   └── app.py                                       # Aplicativo web interactivo en Streamlit (Fase 4)
├── reports/
│   ├── Fase1_Informe_EDA.pdf                        # Entregable escrito de la Fase 1
│   └── Fase4_Informe_Tecnico_Final.pdf              # Informe técnico final en formato artículo
├── requirements.txt                                 # Dependencias para despliegue del aplicativo
└── README.md                                        # Documentación principal del proyecto
```

---

## 6. Cómo ejecutar el proyecto

### Opción A: Ejecutar el tablero interactivo en tu computador

1. Clona este repositorio:
   ```bash
   git clone https://github.com/rmhzo77/proyecto-ciencia-datos-futbol.git
   cd proyecto-ciencia-datos-futbol
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Inicia la aplicación de Streamlit:
   ```bash
   streamlit run dashboard/app.py
   ```

### Opción B: Reproducir todo el análisis y entrenamiento en Google Colab

1. Abre el archivo ubicado en `notebooks/` directamente desde [Google Colab](https://colab.research.google.com/).
2. Ejecuta todas las celdas en orden (**Entorno de ejecución** > **Ejecutar todas**). La primera celda instala automáticamente las librerías analíticas (`scikit-learn`, `torch`, `shap`, `xgboost`, `scipy`, `seaborn`, entre otras) y reproduce de forma determinista el análisis exploratorio, los modelos, las gráficas y las tablas.

---

## 7. Créditos académicos

* **Autor:** Rodrigo Rafael Madera Herazo
* **Programa:** Maestría en Inteligencia Artificial — Universidad de La Salle
* **Asignatura:** Ciencia de Datos
* **Docente:** Fabián Camilo Castro Riveros
* **Fecha:** Noviembre de 2026