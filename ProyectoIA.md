# Universidad ICESI  
**Facultad de Ingeniería, Diseño y Ciencias Aplicadas**  
**Departamento de Computación y Sistemas Inteligentes**  
**Ingeniería de Sistemas**

**Asignatura:** APO3  
**Semestre:** 2025-2

---

## Lineamientos para el proyecto final

El proyecto final del curso **Inteligencia Artificial I** es un trabajo grupal (mínimo 2 y máximo 3 estudiantes por grupo) que busca que los estudiantes desarrollen una solución a un problema real usando modelos de analítica y conjuntos de datos de diferente formato.

Cada grupo debe:

- Entender el problema, investigar su contexto y antecedentes.
- Establecer la metodología de trabajo.
- Proponer métricas de desempeño para evaluar el progreso.
- Entrenar y evaluar diferentes modelos de analítica.
- Ajustar hiperparámetros y evaluar resultados con métricas predefinidas.
- Usar la metodología **CRISP-DM** adaptada a las necesidades del proyecto.

---

## 1. Caso de Estudio Propuesto: Sistema de anotación de video

**Objetivo:**  
Desarrollar una herramienta de software capaz de analizar actividades específicas de una persona (caminar hacia la cámara, caminar de regreso, girar, sentarse, ponerse de pie) y realizar un seguimiento de movimientos articulares y posturales.

**Requerimientos técnicos:**

- **Entradas:** Video en tiempo real capturado por la cámara.
- **Salidas:** Clasificación de la actividad en tiempo real y análisis de inclinaciones laterales y movimientos de articulaciones clave (muñecas, rodillas, caderas).

---

### Recolección de Datos y Anotación

- **Base de Datos:** Captura de videos con varias personas realizando las actividades. Debe incluir diferentes perspectivas y variaciones en velocidades y trayectorias.
- **Anotación:**
  - **Manual:** Etiquetado de segmentos de video con actividades clave.
  - **Automática:** Uso de herramientas como **LabelStudio** o **CVAT**.

---

### Seguimiento de Articulaciones y Movimientos

- **MediaPipe** o **OpenPose** para seguimiento de articulaciones clave.
- **Landmarks a seguir:** Cadera, rodillas, tobillos, muñecas, hombros, cabeza.
- **Inclinación lateral:** Medición a partir de la posición de hombros y caderas.
- **Movimientos:** Estimación de ángulos de articulaciones durante flexión/extensión.

---

### Preprocesamiento de Datos

- **Normalización:** Estandarizar coordenadas de articulaciones.
- **Filtrado:** Suavizado de posiciones para eliminar ruido.
- **Generación de características:**
  - Velocidad de articulaciones.
  - Ángulos relativos entre articulaciones.
  - Inclinación del tronco.

---

### Entrenamiento del Sistema de Clasificación

- **Elección del Modelo:** Modelos supervisados como SVM, Random Forest, XGBoost, etc.
- **Entrenamiento:** División en conjuntos de entrenamiento y prueba.
- **Características:** Posiciones, velocidades, ángulos, etc.

---

### Inferencia en Tiempo Real

- Visualización de la actividad detectada y medidas posturales.
- Interfaz gráfica sencilla para mostrar resultados en tiempo real.

---

### Validación y Evaluación

- Pruebas con varias personas.
- Comparación de predicciones con etiquetas reales.
- Métricas: Precisión, Recall, F1-Score.

---

### Recursos clave

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419)
- [LabelStudio](https://labelstud.io/)
- [CVAT vs LabelStudio](https://medium.com/cvat-ai/cvat-vs-labelstudio-which-one-is-better-bla0d333842e)

---

## 2. Evaluación y entregables

### Criterios de evaluación

- Claridad y robustez de la metodología.
- Aproximaciones razonables en el proyecto.
- Exploración y procesamiento adecuado de datos.
- Soluciones ingeniosas e interesantes.
- Explicación de impactos de la solución en el contexto.
- Desarrollo y transmisión de conocimientos no triviales.
- Demostración de competencias del curso.

---

### Entregables por semana

**Repositorio GitHub con tres carpetas:** `Entrega1`, `Entrega2`, `Entrega3`.

#### 🗓️ Semana 12 – Primera entrega

- Preguntas de interés y tipo de problema.
- Metodología y métricas de progreso.
- Datos recolectados y análisis exploratorio.
- Próximos pasos.
- Estrategias para incrementar el conjunto de datos.
- **Análisis de aspectos éticos** en el contexto de IA.

#### 🗓️ Semana 14 – Segunda entrega

- Estrategia para obtención de nuevos datos.
- Preparación de datos.
- Entrenamiento de modelos y ajuste de hiperparámetros.
- Resultados (métricas, gráficas, etc.).
- Plan de despliegue.
- **Análisis inicial de impactos** de la solución.

#### 🗓️ Semana 17 – Tercera entrega

- Reducción de características.
- Evaluación de resultados.
- Despliegue de la solución.
- Reporte final.
- **Análisis final de impactos** de la solución.
- Video de presentación (máximo 10 minutos).

---

## Aspectos a tener en cuenta

1. **Código fuente bien documentado.**
2. **Referenciar claramente** datos y código de terceros.
3. **Informes claros y concisos**, con diagramas y gráficos de calidad vectorial.

---

## Estructura básica del reporte final (máximo 7 páginas)

1. **Title**
2. **Abstract**
3. **Introduction**: contexto, descripción del problema, por qué es interesante.
4. **Theory**: fundamentos necesarios para entender el desarrollo.
5. **Methodology**: enfoque del proyecto (no copiar diagrama CRISP-DM literal).
6. **Results**: rendimiento de los modelos y métricas de interés.
7. **Results analysis**: observaciones, generalización, sobreajuste, comparación con literatura.
8. **Conclusions and Future Work**: logros, aprendizajes y mejoras posibles.
9. **Bibliographic References**: formato IEEE, solo referencias usadas.

---

**Observación:** Revisar artículos de conferencias como NIPS, ICML o ICLR antes de redactar los reportes finales.

--- 

*Documento convertido a formato Markdown para mejor legibilidad y estructuración.*