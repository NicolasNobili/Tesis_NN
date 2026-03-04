# Tesis FIUBA - Superresolucion Satelital

Repositorio de trabajo de tesis para superresolucion de imagenes satelitales (dataset Sen2Venus y variantes propias), con entrenamiento, evaluacion y analisis de modelos en PyTorch.

## Contenido del repo

- `project_package/`: paquete principal (modelos, datasets, losses, trainer/tester, utilidades).
- `train_scripts/`: scripts de entrenamiento (RCR, RCRAC, RCRAC_MS, UNet, HybridTC, Transformer, SRCNN, etc.).
- `test_scripts/`: scripts de test por modelo.
- `tests_final/`: bateria de pruebas finales y comparativas.
- `aux_scripts/`: generacion de datasets y scripts de visualizacion.
- `datasets/`: datasets locales (no versionados).

## Instalacion

Requisitos recomendados:

- Python 3.10+
- pip actualizado

### 1) Crear y activar entorno virtual

En Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Instalar el paquete del proyecto

```bash
pip install -e .
```

Esto instala `project_package` en modo editable.

### 4) Verificacion rapida

```bash
python -c "import project_package; print('project_package OK')"
```

## Uso rapido

Entrenamiento (ejemplos):

```bash
python train_scripts/Titan_train_RCR_10m.py
python train_scripts/Titan_train_RCRAC_20m.py
```

Test por modelo (ejemplos):

```bash
python test_scripts/test_RCR.py
python test_scripts/test_RCRAC.py
python test_scripts/test_RCRAC_MS.py
```

Bateria de tests finales:

```bash
python tests_final/runt_tests.py
```

## Nota importante

Varios scripts tienen rutas locales/hardcodeadas (por ejemplo `external_ssd`, paths absolutos en Windows o Linux). Antes de correr entrenamiento/test, revisa y ajusta las rutas segun tu entorno y la ubicacion real de `datasets/` y `results/`.
