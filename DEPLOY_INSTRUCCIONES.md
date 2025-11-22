# 🚀 Instrucciones de Despliegue en Railway

## ✅ Archivos Configurados

Tu proyecto ya tiene todo listo para desplegar en Railway:

```
movement-analysis-system-AI/
├── Dockerfile              ✅ Imagen Docker optimizada
├── .dockerignore          ✅ Excluye archivos innecesarios
├── railway.toml           ✅ Configuración de Railway
├── Procfile               ✅ Comando de respaldo
├── runtime.txt            ✅ Python 3.11
│
└── Entrega3/
    ├── app_clasificador_movimiento.py    ← Tu aplicación
    ├── requirements.txt                  ← Dependencias
    └── resultados/                       ← Modelos .pkl (IMPORTANTE)
        ├── random_forest_model.pkl
        ├── svm_model.pkl
        ├── xgboost_model.pkl
        ├── pca_model.pkl
        ├── scaler_minmax.pkl
        └── label_encoder.pkl
```

---

## 📝 Pasos para Desplegar

### 1️⃣ **Verificar que los modelos estén listos**

```bash
# Verifica que existan los archivos .pkl
ls -lh Entrega3/resultados/*.pkl
```

Deberías ver 6 archivos `.pkl`. **IMPORTANTE:** El archivo `svm_model.pkl` debe estar reentrenado con `probability=True`.

---

### 2️⃣ **Subir cambios a GitHub**

```bash
# Ver archivos modificados
git status

# Agregar todos los archivos nuevos
git add Dockerfile .dockerignore railway.toml

# Commit
git commit -m "Deploy: Configuración Docker para Railway"

# Push
git push origin main
```

---

### 3️⃣ **Desplegar en Railway**

1. Ve a [railway.app](https://railway.app/)
2. Crea una cuenta o inicia sesión
3. Haz clic en **"New Project"**
4. Selecciona **"Deploy from GitHub repo"**
5. Autoriza Railway a acceder a tu GitHub
6. Selecciona el repositorio `movement-analysis-system-AI`

Railway detectará automáticamente el `Dockerfile` y comenzará a construir.

---

### 4️⃣ **Ver el progreso del build**

En Railway verás logs como:

```
Building Docker image...
[+] Building Python 3.11 image
[+] Installing system dependencies (OpenCV, MediaPipe)
[+] Installing Python packages from requirements.txt
    - streamlit
    - opencv-python-headless
    - mediapipe
    - scikit-learn
    - xgboost
    - pandas, numpy, matplotlib, seaborn
[+] Copying application files
✓ Build complete (3-5 minutos)

Deploying...
✓ Container started
✓ Health check passed
✓ Deployment successful
```

---

### 5️⃣ **Obtener la URL de tu aplicación**

1. En Railway, ve a **Settings** → **Networking**
2. Haz clic en **"Generate Domain"**
3. Tu aplicación estará disponible en: `https://tu-app.up.railway.app`

---

## 🔧 Solución de Problemas

### ❌ Error: "No module named 'streamlit'"

**Solución:** Verifica que `Entrega3/requirements.txt` esté correcto.

```bash
cat Entrega3/requirements.txt
```

---

### ❌ Error: "FileNotFoundError: resultados/..."

**Problema:** Los archivos `.pkl` no están en el repositorio.

**Solución:**
1. Verifica que la carpeta `Entrega3/resultados/` contenga los 6 archivos `.pkl`
2. Asegúrate de que NO estén en `.gitignore`
3. Haz commit y push:

```bash
git add Entrega3/resultados/*.pkl
git commit -m "Add trained models"
git push origin main
```

---

### ❌ Error: "Application failed to respond"

**Problema:** El puerto no está configurado correctamente.

**Solución:** Ya está resuelto en el `Dockerfile`. Railway asigna el puerto automáticamente usando `$PORT`.

---

### ❌ Build muy lento o se queda atascado

**Problema:** Archivos grandes (videos) en el repositorio.

**Solución:** El archivo `.dockerignore` ya excluye videos. Verifica que no haya archivos `.mp4` en tu repo:

```bash
find . -name "*.mp4" -type f
```

Si encuentra videos, elimínalos del repositorio:

```bash
git rm APO3_EntregaFinal/Entrega2/videos/*.mp4
git commit -m "Remove large video files"
git push origin main
```

---

### ⚠️ Warning: "Out of memory"

**Problema:** Los modelos `.pkl` son muy grandes para Railway Free Tier (512 MB RAM).

**Soluciones:**
1. **Railway Pro** ($5/mes) - 8 GB RAM
2. **Render.com** - Alternativa gratuita con más RAM
3. **Streamlit Cloud** - Gratis, optimizado para apps Streamlit

---

## 📊 Tamaño de los Archivos

Verifica el tamaño de los modelos:

```bash
du -sh Entrega3/resultados/
```

**Recomendaciones:**
- ✅ **< 100 MB total**: Railway Free funciona bien
- ⚠️ **100-200 MB**: Puede funcionar pero será lento
- ❌ **> 200 MB**: Necesitas Railway Pro o alternativa

---

## 🎨 Alternativas a Railway

### **Streamlit Cloud** (Recomendado para apps Streamlit)

1. Ve a [share.streamlit.io](https://share.streamlit.io/)
2. Conecta tu GitHub
3. Selecciona tu repositorio
4. Main file path: `Entrega3/app_clasificador_movimiento.py`
5. ¡Deploy automático!

**Ventajas:**
- ✅ Gratis
- ✅ Optimizado para Streamlit
- ✅ 1 GB RAM (más que Railway Free)
- ✅ Sin configuración de Docker necesaria

---

### **Render.com**

Similar a Railway pero con más recursos en Free Tier.

1. Ve a [render.com](https://render.com/)
2. New → Web Service
3. Connect GitHub repo
4. Build Command: `pip install -r Entrega3/requirements.txt`
5. Start Command: `cd Entrega3 && streamlit run app_clasificador_movimiento.py --server.port $PORT --server.address 0.0.0.0`

---

## ✅ Checklist Pre-Deploy

Antes de hacer push, verifica:

- [ ] ✅ 6 archivos `.pkl` en `Entrega3/resultados/`
- [ ] ✅ SVM con `probability=True` (reentrenado)
- [ ] ✅ `Dockerfile` en la raíz del repo
- [ ] ✅ `.dockerignore` configurado
- [ ] ✅ App funciona localmente
- [ ] ✅ Sin videos `.mp4` en el repo
- [ ] ✅ Tamaño total de modelos < 200 MB

---

## 🧪 Probar Localmente con Docker

Antes de desplegar, puedes probar el Dockerfile localmente:

```bash
# Construir imagen
docker build -t movement-classifier .

# Ejecutar contenedor
docker run -p 8501:8501 movement-classifier

# Abrir en navegador
http://localhost:8501
```

---

## 📞 Soporte

Si tienes problemas:

1. **Logs de Railway:** Revisa los logs en tiempo real
2. **Documentación Railway:** [docs.railway.app](https://docs.railway.app/)
3. **Discord Railway:** Comunidad activa

---

## 🎉 Deploy Exitoso

Una vez desplegado, verás:

```
✓ Build completed
✓ Container running
✓ Health checks passing

🌐 Your app is live at:
https://movement-classifier-production.up.railway.app
```

**¡Comparte tu app!** 🚀

