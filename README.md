# **Sistema de Análisis de Movimiento Humano con IA**

## **Proyecto Final APO 3 - Inteligencia Artificial I**

**Universidad ICESI**  
Facultad de Ingeniería, Diseño y Ciencias Aplicadas  
Departamento de Computación y Sistemas Inteligentes

**Integrantes del grupo:**

* Mariana De La Cruz - A00399618
* Valentina Gómez - A00398790
* Alexis Delgado - A00399176
* Juan Camilo Amorocho - A00399789

---

### **Descripción del proyecto**

El repositorio **`movement-analysis-system-AI`** contiene el desarrollo completo del proyecto final del curso **APO 3**, cuyo objetivo es construir un sistema automatizado para el análisis y clasificación de actividades humanas a partir de video, integrando visión por computadora, aprendizaje automático y análisis biomecánico.

El sistema utiliza **MediaPipe Pose**, que permite identificar 33 puntos de referencia corporales (landmarks). A partir de estos, se extraen 12 métricas biomecánicas como brillo, movimiento, velocidad de cadera, ángulos articulares (rodilla, cadera, tobillo) e inclinación de hombros, con el fin de evaluar la postura y clasificar acciones básicas como caminar, sentarse, agacharse o girar.

Estas métricas son utilizadas por modelos de **Machine Learning (Random Forest, SVM y XGBoost)** para clasificar distintas posturas y acciones humanas, logrando un sistema capaz de detectar automáticamente el tipo de movimiento a partir de la información biomecánica derivada del video.

**Resultados principales:**
- ✅ **94.44% accuracy** con Random Forest y SVM (usando PCA)
- ✅ **Reducción dimensional:** 12 características → 6 componentes principales (91.73% varianza conservada)
- ✅ **Aplicación web funcional** desplegada con Streamlit
- ✅ **8 categorías de actividades** clasificadas automáticamente



## **Estructura del repositorio**

```
movement-analysis-system-AI/
│
├── README.md                      → Descripción general del proyecto
├── ProyectoIA.md                  → Lineamientos del proyecto
├── Informe necesario.md           → Competencias y evidencias requeridas
├── informe_final.md               → Reporte final completo (7 páginas)
├── Dockerfile                     → Configuración para despliegue en Railway
├── .dockerignore                  → Archivos excluidos del build Docker
│
├── APO3_EntregaFinal/
│   ├── Entrega1/                  → Fase inicial del proyecto
│   │   ├── videos/                → Videos originales por categoría
│   │   ├── procesados/            → Videos con esqueleto superpuesto
│   │   ├── landmarks/             → Coordenadas corporales (CSV)
│   │   └── resultados/            → Métricas y reportes
│   │
│   └── Entrega2/                  → Fase de modelado y entrenamiento
│       ├── videos/                → Nuevos videos de entrenamiento
│       ├── procesados/            → Visualización de poses detectadas
│       ├── landmarks/             → Landmarks extraídos (33 joints)
│       └── resultados/            → Datasets limpios, normalizados y métricas
│
├── Entrega1/
│   └── Entrega1_ProyectoFinal_APO3_MovementAnalysis.ipynb
│
├── Entrega2/
│   ├── Entrega2_ProyectoFinal_APO3_MovementAnalysis.ipynb
│   └── entrega2_proyectofinal_apo3_movementanalysis (1).py
│
└── Entrega3/                      → Fase final: Reducción dimensional y despliegue
    ├── app_clasificador_movimiento.py    → Aplicación web Streamlit
    ├── entrega_proyectofinal_apo3_movementanalysis.py  → Script principal
    ├── guardar_modelos.py               → Script para exportar modelos
    ├── reentrenar_svm_con_probabilidades.py
    ├── requirements.txt                 → Dependencias Python
    ├── README_APP.md                    → Guía de uso de la aplicación
    ├── resultados/                    → Modelos entrenados (.pkl)
    │   ├── random_forest_model.pkl
    │   ├── svm_model.pkl
    │   ├── xgboost_model.pkl
    │   ├── pca_model.pkl
    │   ├── scaler_minmax.pkl
    │   ├── label_encoder.pkl
    │   └── [visualizaciones y reportes]
    └── datasets/
        └── dataset_reducido_pca.csv
```

---

## **Fases del proyecto**

### **Entrega 1 — Procesamiento y análisis inicial** (Semana 12)

El notebook `Entrega1_ProyectoFinal_APO3_MovementAnalysis.ipynb` incluye:

* Contexto, objetivos y metodología del proyecto.
* Extracción de métricas visuales sin landmarks (brillo, movimiento, duración, FPS).
* Implementación inicial de **MediaPipe Pose** para detección corporal.
* Generación de reportes y métricas descriptivas por categoría de acción.
* Análisis exploratorio básico (EDA) y visualizaciones comparativas.
* Reflexión ética sobre el uso responsable de la visión por computadora.

**Dataset inicial:** 54 videos (3 categorías: Adelante, Atrás, Sentado)

Resultados disponibles en: `APO3_EntregaFinal/Entrega1/resultados/`

---

### **Entrega 2 — Normalización, modelado y clasificación** (Semana 14)

El notebook `Entrega2_ProyectoFinal_APO3_MovementAnalysis.ipynb` profundiza en la segunda etapa del proyecto, centrada en la creación del modelo de clasificación inteligente.

Incluye:

1. **Estrategia de ampliación de datos:** incorporación de nuevas categorías y ángulos (caderas, lado, sentadillas, tijeras).
2. **Preparación del dataset:**
   * Limpieza de datos y eliminación de columnas irrelevantes (`video`, `resolución`, `fps`).
   * Detección y manejo de outliers.
   * Normalización con **MinMaxScaler**.
3. **Análisis estadístico y correlacional:**
   * Matriz de correlación y visualización con mapa de calor (`sns.heatmap`).
   * Análisis de distribución y relación entre métricas biomecánicas.
4. **Entrenamiento de modelos:**
   * Implementación de **Random Forest**, **SVM (RBF)** y **XGBoost**.
   * Ajuste de hiperparámetros con **GridSearchCV**.
5. **Evaluación comparativa:**
   * Métricas de *accuracy*, *precision*, *recall* y *F1-score*.
   * Visualización de matrices de confusión.
   * Comparación gráfica del rendimiento de cada modelo.
6. **Exportación de resultados y modelos:**
   * Guardado de datasets (`dataset_limpio.csv`, `dataset_normalizado.csv`).

**Dataset ampliado:** 86 videos (8 categorías)

**Resultados:**
- Random Forest: 100% accuracy (overfitting detectado)
- SVM (RBF): 100% accuracy (overfitting detectado)
- XGBoost: 88.9% accuracy

Resultados disponibles en: `APO3_EntregaFinal/Entrega2/resultados/`

---

### **Entrega 3 — Reducción dimensional y despliegue** (Semana 17)

El script `entrega_proyectofinal_apo3_movementanalysis.py` y la aplicación `app_clasificador_movimiento.py` completan el proyecto con:

1. **Reducción dimensional con PCA:**
   * Reducción de 12 características a 6 componentes principales
   * Conservación del 91.73% de varianza
   * Mitigación de overfitting observado en Entrega 2

2. **Reentrenamiento de modelos con PCA:**
   * Random Forest: **94.44% accuracy** (mejor generalización)
   * SVM (RBF): **94.44% accuracy** (mejor generalización)
   * XGBoost: 77.78% accuracy

3. **Aplicación web interactiva:**
   * Interfaz Streamlit para clasificación en tiempo real
   * Carga de videos (MP4, AVI, MOV)
   * Visualización de métricas biomecánicas
   * Análisis de componentes principales
   * Descarga de resultados en CSV

4. **Despliegue en producción:**
   * Configuración Docker para Railway
   * Documentación de despliegue completa

5. **Reporte final:**
   * Análisis completo de resultados
   * Evaluación de competencias (P11, P12, P13)
   * Análisis ético y de impactos
   * Formulación matemática del problema

**Archivos principales:**
- `Entrega3/entrega_proyectofinal_apo3_movementanalysis.py` - Script de análisis completo
- `Entrega3/app_clasificador_movimiento.py` - Aplicación web Streamlit
- `Entrega3/resultados/` - Modelos entrenados y visualizaciones
- `informe_final.md` - Reporte final completo

---

## **🚀 Instalación y Uso**

### **Requisitos previos**

- Python 3.11+
- pip
- Git

### **Instalación local**

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/movement-analysis-system-AI.git
cd movement-analysis-system-AI

# 2. Instalar dependencias
cd Entrega3
pip install -r requirements.txt

# 3. Verificar que los modelos estén presentes
ls resultados/*.pkl
# Deben existir: random_forest_model.pkl, svm_model.pkl, xgboost_model.pkl,
#                pca_model.pkl, scaler_minmax.pkl, label_encoder.pkl

# 4. Ejecutar la aplicación
streamlit run app_clasificador_movimiento.py
```

La aplicación se abrirá en `http://localhost:8501`

### **Despliegue en Railway**

1. **Preparar el repositorio:**
   ```bash
   # Asegúrate de tener todos los modelos en Entrega3/resultados/
   git add Entrega3/resultados/*.pkl
   git commit -m "Add trained models"
   git push origin main
   ```

2. **Configurar Railway:**
   - Ve a [railway.app](https://railway.app/)
   - Crea un nuevo proyecto desde GitHub
   - Selecciona este repositorio
   - Railway detectará automáticamente el `Dockerfile`

3. **Verificar configuración:**
   - Settings → Build: Debe decir "Dockerfile"
   - Settings → Deploy: No debe haber "Custom Start Command"
   - Settings → Variables: Railway asigna `PORT` automáticamente

4. **Desplegar:**
   - Railway construirá la imagen Docker automáticamente
   - El despliegue tomará ~3-5 minutos
   - Obtén la URL en Settings → Networking

**Documentación detallada:** Ver `DEPLOY_INSTRUCCIONES.md` y `Entrega3/README_APP.md`

---

## **📊 Resultados y Métricas**

### **Rendimiento de Modelos (Entrega 3 - con PCA)**

| Modelo | Accuracy | Precision | Recall | F1-Score | CV Score |
|--------|----------|-----------|--------|----------|----------|
| Random Forest | 94.44% | 95.56% | 94.44% | 94.36% | 98.46% |
| SVM (RBF) | 94.44% | 95.56% | 94.44% | 94.36% | 100.0% |
| XGBoost | 77.78% | 76.85% | 77.78% | 75.45% | 97.03% |

### **Análisis PCA**

- **Características originales:** 12
- **Componentes principales:** 6
- **Reducción dimensional:** 50%
- **Varianza conservada:** 91.73%

### **Actividades Clasificadas**

1. **Adelante** - Caminar hacia la cámara
2. **Atrás** - Caminar alejándose
3. **Sentado** - Posición sentada
4. **Cadera al frente** - Flexión de cadera frontal
5. **Caderas** - Rotación de caderas
6. **Lado** - Movimiento lateral
7. **Sentadilla** - Sentadilla profunda
8. **Tijeras** - Movimiento de tijeras

---

## **📚 Documentación Adicional**

- **`ProyectoIA.md`** - Lineamientos y requisitos del proyecto
- **`Informe necesario.md`** - Competencias y evidencias requeridas
- **`informe_final.md`** - Reporte final completo (7 páginas)
- **`Entrega3/README_APP.md`** - Guía detallada de la aplicación web
- **`DEPLOY_INSTRUCCIONES.md`** - Instrucciones de despliegue en Railway

---

## **🛠️ Tecnologías Utilizadas**

- **MediaPipe Pose** - Detección de landmarks corporales (33 puntos)
- **OpenCV** - Procesamiento de video
- **Scikit-learn** - Random Forest, SVM, PCA, MinMaxScaler
- **XGBoost** - Gradient Boosting
- **Streamlit** - Interfaz web interactiva
- **Pandas, NumPy** - Manipulación de datos
- **Matplotlib, Seaborn** - Visualización
- **Docker** - Containerización para despliegue
- **Railway** - Plataforma de despliegue

---

## **📝 Notas Importantes**

### **Limitaciones del Sistema**

- **Dataset pequeño:** 86 videos limita la generalización
- **Sobreajuste inicial:** Detectado en Entrega 2 (100% accuracy)
- **Validación externa:** No probado con personas diferentes al conjunto de entrenamiento
- **Condiciones controladas:** Mejor rendimiento con iluminación y fondo uniformes

### **Recomendaciones para Mejora**

- Recolectar más datos (objetivo: 200+ videos por categoría)
- Validar con personas diferentes al conjunto de entrenamiento
- Implementar modelado temporal (LSTM, GRU)
- Explorar Graph Neural Networks para estructura esquelética
- Realizar estudios clínicos para validación médica

---

## **👥 Equipo de Desarrollo**

**Proyecto Final APO 3 - Inteligencia Artificial I**  
**Semestre 2025-2**

- Mariana De La Cruz - A00399618
- Valentina Gómez - A00398790
- Alexis Delgado - A00399176
- Juan Camilo Amorocho - A00399789

---

## **📜 Licencia**

Este proyecto es parte del curso APO 3 y es de uso académico.

---

## **🎓 Referencias**

- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Railway Documentation](https://docs.railway.app/)

---

**Última actualización:** Enero 2025
