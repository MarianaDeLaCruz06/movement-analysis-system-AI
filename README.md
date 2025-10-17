# **Entrega 1 - Proyecto Final APO 3**

## **Sistema de Anotación de Video para Análisis de Actividades Humanas**

**Integrantes del grupo:**

*   Mariana De La Cruz - A00399618
*   Valentina Gómez - A00398790
*   Alexis Delgado - A00399176
*   Juan Camilo Amorocho - A00399789

### **Descripción del proyecto:**
Este proyecto implementa un sistema de análisis automático de actividades humanas a partir de videos capturados con la cámara de un teléfono. Utilizando MediaPipe y técnicas de visión por computadora, el sistema procesa los movimientos del cuerpo para extraer métricas como brillo, movimiento, velocidad de cadera, ángulos articulares e inclinación de hombros, con el fin de evaluar la postura y clasificar acciones básicas como caminar, sentarse o ponerse de pie.


### **Contenido del proyecto**

El archivo **`Entrega1_ProyectoFinal_APO3_MovementAnalysis.ipynb`** contiene todo el desarrollo de la primera entrega del proyecto final de APO 3, centrado en la creación de un sistema de análisis de movimiento humano basado en visión por computadora.
Dentro del notebook se encuentran los apartados de antecedentes, contexto, descripción del proceso, pregunta de interés, objetivos generales y específicos, tipo de problema, metodología, métricas utilizadas, datos recolectados, análisis exploratorio, resultados comparativos y conclusiones, junto con una reflexión ética sobre el uso responsable de sistemas de visión artificial.


El proyecto aborda dos fases principales:


1. Un **análisis inicial sin landmarks**, enfocado en métricas generales del video como brillo, movimiento y duración.
2. Un **análisis avanzado con landmarks**, implementado mediante **MediaPipe Pose**, donde se extraen coordenadas articulares y se calculan métricas biomecánicas (inclinación de hombros, velocidad de cadera, ángulo de rodilla, etc.) para estudiar el comportamiento corporal con mayor precisión.


Todos los resultados y materiales se organizan en la carpeta **`APO3_EntregaFinal`**, que incluye:

📂 **videos/** — grabaciones originales realizadas con cámara RGB (teléfono móvil).

📂 **procesados/** — videos con el esqueleto 3D superpuesto y análisis visual de pose.

📂 **landmarks/** — archivos CSV con las coordenadas de las 33 articulaciones detectadas por frame.

📂 **resultados/** — reportes estadísticos, métricas globales y visualizaciones generadas durante el análisis.


