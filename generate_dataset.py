import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Generate synthetic data
n_samples = 10000
data = {
    'Severity': np.random.randint(1, 5, n_samples),
    'Temperature(F)': np.random.uniform(-20, 120, n_samples),
    'Humidity(%)': np.random.uniform físicas para hacer predicciones.
- **Visualización**: Gráfico de importancia de características para mostrar qué factores influyen más en la severidad.

## Instrucciones de Configuración

1. **Clonar el Repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/traffic-accident-prediction.git
   cd traffic-accident-prediction
   ```

2. **Instalar Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generar el Conjunto de Datos** (si falta `accidents.csv`):
   ```bash
   python generate_dataset.py
   ```

4. **Generar el Modelo**:
   ```bash
   python train_model.py
   ```

5. **Ejecutar la Aplicación Localmente**:
   ```bash
   streamlit run app.py
   ```

## Pruebas en Google Colab

1. Sube `app.py`, `train_model.py`, `generate_dataset.py` y (opcionalmente) `accidents.csv` a Colab.
2. Instala las dependencias:
   ```bash
   !pip install streamlit>=1.20.0 scikit-learn>=1.2.0 pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.6.0 seaborn>=0.12.0 joblib>=1.2.0 pyngrok
   ```
3. Genera el conjunto de datos (si es necesario):
   ```bash
   !python generate_dataset.py
   ```
4. Genera el modelo:
   ```bash
   !python train_model.py
   ```
5. Configura ngrok:
   ```bash
   !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
   !unzip ngrok-stable-linux-amd64.zip
   !./ngrok authtoken TU_AUTH_TOKEN
   ```
6. Ejecuta la aplicación:
   ```bash
   !streamlit run app.py &>/dev/null &
   !curl -s http://localhost:4040/api/tunnels | python3 -c \
       'import sys, json; print("Public URL:", json.load(sys.stdin)["tunnels"][0]["public_url"])'
   ```

## Despliegue

Despliega en [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Inicia sesión con GitHub.
2. Crea una nueva aplicación, selecciona este repositorio y establece `app.py` como el archivo principal.
3. Despliega para obtener una URL pública.

## Licencia

Licencia MIT (ver archivo `LICENSE`).

---

Este proyecto es ideal para un proyecto universitario, ya que es simple, funcional y muestra habilidades en aprendizaje automático y desarrollo web. Si necesitas ajustar características o tienes más requisitos, ¡avísame!
