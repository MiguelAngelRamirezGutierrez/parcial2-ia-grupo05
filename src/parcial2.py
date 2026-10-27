"""
Parcial 2: Inteligencia Artificial - Perceptrón y Calidad de Datos
Docente: Jhon James Cano Sánchez
Estudiante: Miguel Ángel Ramírez Gutiérrez
Grupo: 05 - Dataset: Calidad del aire (Cartago, Valle del Cauca)
Variables: pm25_ugm3 (x1), viento_m_s (x2) | Objetivo: alerta_calidad (y)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo visual para gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR = BASE_DIR  # Guardar los gráficos en la raíz del repositorio

RAW_CSV_PATH = os.path.join(DATA_DIR, 'grupo_05.csv')
CLEAN_CSV_PATH = os.path.join(DATA_DIR, 'grupo_05_limpio.csv')

print("=" * 80)
print("PARCIAL 2: INTELIGENCIA ARTIFICIAL - GRUPO 05")
print("Estudiante: Miguel Ángel Ramírez Gutiérrez")
print("Problema: Calidad del aire en Cartago (Alerta Ambiental)")
print("=" * 80)


# ==============================================================================
# TAREA 2: CARGA E INSPECCIÓN DE DATOS
# ==============================================================================
print("\n" + "=" * 80)
print(">>> TAREA 2: CARGA E INSPECCIÓN INICIAL DE DATOS")
print("=" * 80)

# Carga del CSV original
df_raw = pd.read_csv(RAW_CSV_PATH)

print("\n1. Primeras 5 filas del dataset original:")
print(df_raw.head())

print(f"\n2. Forma del DataFrame (shape): {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

print("\n3. Tipos de datos por columna (dtypes):")
print(df_raw.dtypes)

print("\n4. Cantidad de valores nulos por columna:")
print(df_raw.isnull().sum())

print(f"\n5. Cantidad de filas duplicadas: {df_raw.duplicated().sum()}")

print("\n[PREGUNTA CLAVE TAREA 2]:")
print("¿Alguna columna numérica aparece con un tipo de dato que no debería tener? ¿Por qué ocurre?")
print("-> RESPUESTA: Sí, la columna 'viento_m_s' aparece con tipo 'object' (texto) en lugar de 'float64'.")
print("   Esto ocurre porque el sistema exportó algunos valores usando la coma ',' como separador decimal")
print("   (por ejemplo: '2,0', '7,0', '5,6') en lugar del punto estándar. Al detectar cadenas con coma,")
print("   el parser de Pandas clasifica toda la columna como tipo object.")


# ==============================================================================
# TAREA 3: CALIDAD Y LIMPIEZA DE DATOS CON PANDAS
# ==============================================================================
print("\n" + "=" * 80)
print(">>> TAREA 3: CALIDAD Y LIMPIEZA DE DATOS CON PANDAS")
print("=" * 80)

df_clean = df_raw.copy()

# Problema 1: Filas duplicadas
n_duplicados = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates().copy()
print(f"1. Filas duplicadas eliminadas: {n_duplicados}")

# Problema 2: Formato decimal con comas en viento_m_s
# Convertir a string, reemplazar comas por puntos y transformar a float
df_clean['viento_m_s'] = df_clean['viento_m_s'].astype(str).str.replace(',', '.')
df_clean['viento_m_s'] = pd.to_numeric(df_clean['viento_m_s'], errors='coerce')
print("2. Columna 'viento_m_s' corregida: comas reemplazadas por puntos y convertida a float64.")

# Problema 4 (Pregunta clave): Outlier / valor imposible en pm25_ugm3
media_pm25_antes = df_clean['pm25_ugm3'].mean()
std_pm25_antes = df_clean['pm25_ugm3'].std()
max_pm25_antes = df_clean['pm25_ugm3'].max()

# Fila M075 tiene 70100.0 ug/m3 (error de factor 1000, debería ser 70.1 ug/m3)
df_clean['pm25_ugm3'] = df_clean['pm25_ugm3'].replace(70100.0, 70.1)

media_pm25_despues = df_clean['pm25_ugm3'].mean()
std_pm25_despues = df_clean['pm25_ugm3'].std()

print("\n[PREGUNTA CLAVE TAREA 3 - VALOR SOSPECHOSO]:")
print(f"-> Valor más sospechoso detectado: {max_pm25_antes} µg/m³ en la medición M075.")
print(f"   Media de 'pm25_ugm3' ANTES de corregir: {media_pm25_antes:.2f} µg/m³ (Desviación: {std_pm25_antes:.2f})")
print(f"   Media de 'pm25_ugm3' DESPUÉS de corregir: {media_pm25_despues:.2f} µg/m³ (Desviación: {std_pm25_despues:.2f})")
print("   Explicación del impacto: Un valor de 70,100 µg/m³ es físicamente absurdo e imposible en una ciudad.")
print("   Al estandarizar (X - media) / std, la enorme desviación distorsiona completamente la escala, comprimiendo")
print("   todos los demás 79 registros en valores cercanos a z ≈ -0.1, impidiendo que el perceptrón aprenda la frontera.")

# Problema 3: Valores faltantes (NaN) en viento_m_s
n_nulos_viento = df_clean['viento_m_s'].isnull().sum()
df_clean = df_clean.dropna().copy()
print(f"\n3. Registros con valores nulos eliminados en 'viento_m_s': {n_nulos_viento}")

# Guardar CSV depurado
df_clean.to_csv(CLEAN_CSV_PATH, index=False)
print(f"\n-> Dataset depurado guardado exitosamente en: {CLEAN_CSV_PATH}")

print(f"\nBalance del dataset tras la limpieza:")
print(f"• Filas antes de la limpieza: {len(df_raw)}")
print(f"• Filas después de la limpieza: {len(df_clean)}")
print(f"• Distribución de la variable objetivo (alerta_calidad):")
conteo_clases = df_clean['alerta_calidad'].value_counts()
print(f"  - Clase 0 (Sin alerta): {conteo_clases[0]} registros ({conteo_clases[0]/len(df_clean)*100:.1f}%)")
print(f"  - Clase 1 (Alerta ambiental): {conteo_clases[1]} registros ({conteo_clases[1]/len(df_clean)*100:.1f}%)")


# ==============================================================================
# TAREA 4: VISUALIZACIÓN CON SEABORN
# ==============================================================================
print("\n" + "=" * 80)
print(">>> TAREA 4: VISUALIZACIÓN EXPLORATORIA CON SEABORN")
print("=" * 80)

# Gráfico 1: Dispersión coloreada por clase
plt.figure(figsize=(9, 6))
palette_colors = {0: '#2563eb', 1: '#dc2626'}
scatter = sns.scatterplot(
    data=df_clean,
    x='viento_m_s',
    y='pm25_ugm3',
    hue='alerta_calidad',
    palette=palette_colors,
    style='alerta_calidad',
    s=90,
    alpha=0.9,
    edgecolor='black'
)
plt.title('Gráfico 1: Dispersión de Calidad del Aire en Cartago (PM2.5 vs Viento)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Velocidad del Viento (m/s)', fontsize=11, fontweight='bold')
plt.ylabel('Concentración de PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
plt.axhline(50, color='#f59e0b', linestyle='--', linewidth=1, label='Umbral referencial PM2.5 (50 µg/m³)')

handles, labels = scatter.get_legend_handles_labels()
plt.legend(
    handles=handles[:2],
    labels=['0: Calidad Aceptable (Sin Alerta)', '1: Alerta de Calidad del Aire'],
    title='Estado de Alerta',
    loc='upper right',
    frameon=True
)
plt.tight_layout()
g1_path = os.path.join(IMG_DIR, 'grafico1_dispersion.png')
plt.savefig(g1_path, dpi=300)
plt.close()
print(f"• Gráfico 1 guardado: {g1_path}")

# Gráfico 2: Boxplot comparativo de PM2.5 por estado de alerta
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_clean,
    x='alerta_calidad',
    y='pm25_ugm3',
    palette=['#3b82f6', '#ef4444'],
    width=0.45,
    boxprops=dict(alpha=0.85),
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"}
)
sns.stripplot(
    data=df_clean,
    x='alerta_calidad',
    y='pm25_ugm3',
    color='black',
    alpha=0.4,
    jitter=0.2,
    size=5
)
plt.title('Gráfico 2: Distribución de PM2.5 según Estado de Alerta Ambiental', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Estado de Alerta de Calidad (alerta_calidad)', fontsize=11, fontweight='bold')
plt.ylabel('Concentración de PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
plt.xticks([0, 1], ['0: Sin Alerta', '1: Alerta Ambiental'])
plt.tight_layout()
g2_path = os.path.join(IMG_DIR, 'grafico2_comparacion.png')
plt.savefig(g2_path, dpi=300)
plt.close()
print(f"• Gráfico 2 guardado: {g2_path}")


# ==============================================================================
# TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY
# ==============================================================================
print("\n" + "=" * 80)
print(">>> TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY (CLASE 9)")
print("=" * 80)

# Separación de características X y objetivo y
X_raw = df_clean[['pm25_ugm3', 'viento_m_s']].values.astype(float)
y = df_clean['alerta_calidad'].values.reshape(-1, 1).astype(float)
m = len(y)

# Estandarización de características (Media 0, Desviación 1)
mu = np.mean(X_raw, axis=0)
sigma = np.std(X_raw, axis=0)
X = (X_raw - mu) / sigma

print(f"• Parámetros de estandarización calculados:")
print(f"  - pm25_ugm3: media = {mu[0]:.4f}, std = {sigma[0]:.4f}")
print(f"  - viento_m_s: media = {mu[1]:.4f}, std = {sigma[1]:.4f}")

# Funciones del Perceptrón Sigmoide
def sigmoid(z):
    """Función de activación sigmoide."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def predict_proba(X_mat, w_vec, b_scalar):
    """Calcula la probabilidad predicha p = sigmoid(X @ w + b)."""
    return sigmoid(X_mat @ w_vec + b_scalar)

def compute_loss(y_true, y_prob):
    """Pérdida logarítmica (Binary Cross-Entropy)."""
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_prob_clipped) + (1.0 - y_true) * np.log(1.0 - y_prob_clipped))

def compute_accuracy(y_true, y_prob):
    """Exactitud de clasificación con umbral 0.5."""
    predictions = (y_prob >= 0.5).astype(float)
    return np.mean(predictions == y_true)

# Inicialización de pesos y sesgo (reproducible)
np.random.seed(42)
w = np.random.randn(2, 1) * 0.01
b = 0.0

w_inicial = w.copy()
b_inicial = b

# Parámetros de entrenamiento
epochs = 500
learning_rate = 0.1

history_loss = []
history_acc = []

# Pérdida inicial
prob_inicial = predict_proba(X, w, b)
loss_inicial = compute_loss(y, prob_inicial)
acc_inicial = compute_accuracy(y, prob_inicial)

# Bucle de Descenso del Gradiente
for epoch in range(epochs):
    # Forward pass
    p = predict_proba(X, w, b)
    
    # Métricas
    loss_val = compute_loss(y, p)
    acc_val = compute_accuracy(y, p)
    
    history_loss.append(loss_val)
    history_acc.append(acc_val)
    
    # Backward pass (Gradientes)
    error = p - y
    dw = (X.T @ error) / m
    db = np.mean(error)
    
    # Actualización de parámetros
    w -= learning_rate * dw
    b -= learning_rate * db

# Métricas finales
prob_final = predict_proba(X, w, b)
loss_final = compute_loss(y, prob_final)
acc_final = compute_accuracy(y, prob_final)

print("\nResultados del Entrenamiento:")
print(f"• Pesos iniciales: w1 = {w_inicial[0,0]:.4f}, w2 = {w_inicial[1,0]:.4f}, b = {b_inicial:.4f}")
print(f"• Pérdida inicial: {loss_inicial:.4f} | Exactitud inicial: {acc_inicial*100:.2f}%")
print(f"• Pesos finales:   w1 = {w[0,0]:.4f}, w2 = {w[1,0]:.4f}, b = {b:.4f}")
print(f"• Pérdida final:   {loss_final:.4f} | Exactitud final:   {acc_final*100:.2f}%")

# Gráfico 3: Curva de aprendizaje (Pérdida y Exactitud por época)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

line1 = ax1.plot(range(1, epochs + 1), history_loss, color='#dc2626', linewidth=2, label='Pérdida Logarítmica')
line2 = ax2.plot(range(1, epochs + 1), [acc * 100 for acc in history_acc], color='#2563eb', linewidth=2, linestyle='--', label='Exactitud (%)')

ax1.set_xlabel('Época de Entrenamiento', fontsize=11, fontweight='bold')
ax1.set_ylabel('Pérdida (Binary Cross-Entropy)', color='#dc2626', fontsize=11, fontweight='bold')
ax2.set_ylabel('Exactitud (%)', color='#2563eb', fontsize=11, fontweight='bold')
ax1.set_title('Curva de Aprendizaje del Perceptrón (Descenso del Gradiente)', fontsize=13, fontweight='bold', pad=12)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True)

plt.tight_layout()
curva_path = os.path.join(IMG_DIR, 'curva_aprendizaje.png')
plt.savefig(curva_path, dpi=300)
plt.close()
print(f"• Curva de aprendizaje guardada: {curva_path}")

# Gráfico 4: Frontera de decisión en unidades originales
# z = w1 * ((x1 - mu1) / sigma1) + w2 * ((x2 - mu2) / sigma2) + b = 0
# Despejamos x1 (pm25) en función de x2 (viento):
# x1 = mu1 - sigma1 * (w2 * ((x2 - mu2) / sigma2) + b) / w1

viento_range = np.linspace(X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5, 200)
viento_norm = (viento_range - mu[1]) / sigma[1]
pm25_boundary_norm = -(w[1, 0] * viento_norm + b) / w[0, 0]
pm25_boundary = pm25_boundary_norm * sigma[0] + mu[0]

plt.figure(figsize=(9, 6))
# Graficar puntos de datos reales
sns.scatterplot(
    data=df_clean,
    x='viento_m_s',
    y='pm25_ugm3',
    hue='alerta_calidad',
    palette={0: '#2563eb', 1: '#dc2626'},
    s=80,
    alpha=0.9,
    edgecolor='black'
)
# Graficar la línea de la frontera
plt.plot(viento_range, pm25_boundary, color='#0f172a', linewidth=2.5, linestyle='-', label='Frontera de Decisión (p=0.5)')

# Limitar área visible a valores físicamente razonables
plt.xlim(X_raw[:, 1].min() - 0.3, X_raw[:, 1].max() + 0.3)
plt.ylim(0, X_raw[:, 0].max() + 8)

plt.title('Frontera de Decisión del Perceptrón en Unidades Originales', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Velocidad del Viento (m/s)', fontsize=11, fontweight='bold')
plt.ylabel('Concentración de PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()

frontera_path = os.path.join(IMG_DIR, 'frontera_decision.png')
plt.savefig(frontera_path, dpi=300)
plt.close()
print(f"• Frontera de decisión guardada: {frontera_path}")


# ==============================================================================
# TAREA 6: INTERPRETACIÓN Y EVALUACIÓN DE CASOS NUEVOS
# ==============================================================================
print("\n" + "=" * 80)
print(">>> TAREA 6: INTERPRETACIÓN DE RESULTADOS Y CASOS DE PRUEBA")
print("=" * 80)

# Registro 1: Alta contaminación (75 ug/m3) y viento muy bajo (1.2 m/s) -> Se espera Alerta = 1
caso_1 = np.array([[75.0, 1.2]])
caso_1_norm = (caso_1 - mu) / sigma
prob_1 = predict_proba(caso_1_norm, w, b)[0, 0]

# Registro 2: Aire limpio (15 ug/m3) y viento fuerte (7.5 m/s) -> Se espera Alerta = 0
caso_2 = np.array([[15.0, 7.5]])
caso_2_norm = (caso_2 - mu) / sigma
prob_2 = predict_proba(caso_2_norm, w, b)[0, 0]

print(f"\n1. Registro de Prueba 1: PM2.5 = 75.0 µg/m³, Viento = 1.2 m/s")
print(f"   -> Probabilidad de Alerta: {prob_1 * 100:.2f}% | Clasificación: {'1 (ALERTA AMBIENTAL)' if prob_1 >= 0.5 else '0 (NORMAL)'}")

print(f"\n2. Registro de Prueba 2: PM2.5 = 15.0 µg/m³, Viento = 7.5 m/s")
print(f"   -> Probabilidad de Alerta: {prob_2 * 100:.2f}% | Clasificación: {'1 (ALERTA AMBIENTAL)' if prob_2 >= 0.5 else '0 (NORMAL)'}")

# Registros clasificados incorrectamente
predicciones_totales = (prob_final >= 0.5).astype(float)
lado_equivocado = np.sum(predicciones_totales != y)
print(f"\n3. Registros que quedan del lado equivocado de la frontera: {lado_equivocado} de {m} ({lado_equivocado/m*100:.1f}%)")

print("\n" + "=" * 80)
print("¡EJECUCIÓN DEL SCRIPT PARCIAL2.PY COMPLETADA CON ÉXITO!")
print("=" * 80)
