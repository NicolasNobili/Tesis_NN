# Tesis de Grado - Ingeniería Electrónica (FIUBA)  
## Superresolución de Imágenes Satelitales con Deep Learning  

Este repositorio contiene el desarrollo de la tesis de grado en Ingeniería Electrónica (FIUBA), cuyo objetivo es investigar e implementar distintos métodos de **superresolución de imágenes satelitales**, con foco en datos del dataset **Sen2Venus**.  

---

## 🎯 Objetivo  

Evaluar y comparar el desempeño de distintos enfoques basados en **deep learning** para aumentar la resolución espacial de imágenes satelitales multiespectrales.  

El trabajo busca responder preguntas como:  

- ¿Qué arquitectura logra el mejor balance entre precisión y costo computacional?  
- ¿Cómo se comportan los distintos enfoques en diferentes bandas espectrales?  
- ¿Qué ventajas presentan los modelos modernos (transformers, modelos difusivos) frente a las arquitecturas más clásicas (CNN, U-Net)?  

---

## 📊 Dataset  

- **Sen2Venus**: Dataset que combina observaciones de los satélites **Sentinel-2** y **Venus**, utilizado en tareas de superresolución remota.  
- Incluye imágenes multiespectrales en diferentes resoluciones espaciales, lo que permite entrenar y evaluar modelos de **aprendizaje supervisado**.  

---

## 🧠 Métodos Implementados  

Los métodos desarrollados se basan en arquitecturas modernas de **Deep Learning** aplicadas a visión por computadora.  

### 1. **Redes Convolucionales (CNN)**  
- Primer enfoque base para superresolución.  
- Se emplearon arquitecturas profundas con capas convolucionales y activaciones no lineales.  
- Punto de partida para comparar con modelos más complejos.  

### 2. **Redes Residuales (ResNet / SRResNet)**  
- Se introdujeron bloques residuales para mejorar el flujo de gradientes.  
- Inspirado en **SRResNet**, ampliamente utilizado en superresolución de imágenes naturales.  
- Ventaja: permite entrenar redes más profundas sin degradación del rendimiento.  

### 3. **U-Net**  
- Arquitectura encoder-decoder con **skip-connections**.  
- Permite capturar información de bajo nivel (detalles espaciales) y alto nivel (contexto).  
- Ampliamente utilizada en visión satelital y biomédica.  


### 4. **Swin Transformer**  
- Arquitectura basada en **Transformers jerárquicos** con ventanas deslizantes.  
- Capacidad de modelar dependencias a largo alcance en las imágenes satelitales.  
- Promete mejor rendimiento en comparación con CNN clásicas al capturar **relaciones globales entre píxeles**.  

---

## ⚙️ Tecnologías utilizadas  

- **PyTorch**: Framework principal para el desarrollo de los modelos.  
- **NumPy, Pandas, Matplotlib**: Preprocesamiento y visualización de resultados.  
- **scikit-learn**: Métricas de evaluación (PSNR, SSIM).  

## 📈 Evaluación  

Los modelos se evaluaron en función de:  

- **Métricas cuantitativas**:  
  - PSNR (Peak Signal-to-Noise Ratio)  
  - SSIM (Structural Similarity Index)  

- **Métricas cualitativas**:  
  - Inspección visual de detalles reconstruidos.  
  - Evaluación perceptual de la calidad de las imágenes generadas con LPIPS.  

---

## 🚀 Próximos pasos  

- Optimización de hiperparámetros y arquitecturas híbridas.
- Extensión a otros datos multiespectrales.  
