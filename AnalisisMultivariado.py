import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Cargando los datos
ruta = "C:/Users/ben19/Downloads/examen2/NaturalGas.csv"
df = pd.read_csv(ruta)

# Renombrando columnas para facilitar su interpretación
df.columns = ['indice', 'estado', 'codigo_estado', 'año', 'consumo', 'precio_gas', 'precio_electricidad', 
              'precio_petroleo', 'precio_gas_liquido', 'grados_calefaccion', 'ingreso']

# Mostrar información general del dataset
print(df.info())

# Objetivo 1: Analizar la relación entre el consumo de gas natural en el sector residencial y factores económicos y climáticos.
# Excluir columnas no numéricas
print("\nObjetivo 1:")
df_numeric = df.drop(['indice', 'estado', 'codigo_estado', 'año'], axis=1)

# Correlación de Variables
correlacion = df_numeric.corr()
print("\nCorrelación de Variables:")
print(correlacion)

# Generando el modelo de regresión lineal múltiple
print("\nGenerando el modelo de regresión lineal múltiple:")
modelo = smf.ols(formula='consumo ~ precio_gas + precio_electricidad + precio_petroleo + precio_gas_liquido + grados_calefaccion + ingreso', data=df).fit()

# Mostrar resumen del modelo
print("\nResumen del modelo:")
print(modelo.summary())

# Obtener los valores de R, R^2 y R^2 ajustada
r = np.sqrt(modelo.rsquared)
r2 = modelo.rsquared
r2_adj = modelo.rsquared_adj

# Mostrar los valores en consola
print("\nValores de R, R^2 y R^2 ajustada:")
print(f"R: {r}")
print(f"R^2: {r2}")
print(f"R^2 ajustada: {r2_adj}")


# Matriz de mínimos cuadrados
#print("\nMatriz de Mínimos Cuadrados:")
#matriz_mc = modelo.summary().tables[1]
#print(matriz_mc)

# Objetivo 2: Construir modelos de regresión lineal múltiple para predecir el consumo de gas natural basado en las variables seleccionadas.
# Seleccionando los mejores predictores
# Aquí puedes utilizar técnicas más avanzadas como forward/backward stepwise, pero para este ejemplo, usaremos todos los predictores
print("\nObjetivo 2:")
print("\nEl mejor modelo resultante es:")
print(modelo.params)

# Hallando el intervalo de confianza
print("\nIntervalo de confianza:")
print(modelo.conf_int())

# Distribución normal de los residuos (Shapiro-Wilk)
residuos = modelo.resid
shapiro_test = stats.shapiro(residuos)
print("\nDistribución normal de los residuos (Shapiro-Wilk):")
print("Estadístico de prueba:", shapiro_test[0])
print("Valor p:", shapiro_test[1])

# Homocedasticidad (Prueba de Breusch-Pagan)
bp_test = sm.stats.diagnostic.het_breuschpagan(residuos, modelo.model.exog)
print("\nHomocedasticidad (Prueba de Breusch-Pagan):")
print("Estadístico de prueba:", bp_test[0])
print("Valor p:", bp_test[1])

# Matriz de correlación entre predictores
print("\nMatriz de correlación entre predictores:")
matriz_correlacion_predictores = df_numeric.corr()
print(matriz_correlacion_predictores)

# Análisis de Inflación de Varianza (VIF)
print("\nObjetivo 3:")
X = df_numeric[['precio_gas', 'precio_electricidad', 'precio_petroleo', 'precio_gas_liquido', 'grados_calefaccion', 'ingreso']]
vif_data = pd.DataFrame()
vif_data["Predictor"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\nAnálisis de Inflación de Varianza (VIF):")
print(vif_data)

# Autocorrelación (Durbin-Watson)
durbin_watson = sm.stats.stattools.durbin_watson(residuos)
print("\nAutocorrelación (Durbin-Watson):", durbin_watson)

# Identificación de posibles valores atípicos o influyentes
# Atipicidad (Gráficamente)
sns.set(style="whitegrid")
residuos_df = pd.DataFrame({'Fitted': modelo.fittedvalues, 'Residuos': residuos})
sns.residplot(x='Fitted', y='Residuos', data=residuos_df, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1})
plt.axhline(y=0, color='gray', linestyle='--', lw=2)
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")
plt.title("Gráfico de Residuos para Identificar Valores Atípicos o Influyentes")
plt.show()


# Predicciones
predicciones = modelo.predict(df[['precio_gas', 'precio_electricidad', 'precio_petroleo', 'precio_gas_liquido', 'grados_calefaccion', 'ingreso']])
df['predicciones'] = predicciones

# Gráfico de predicciones vs. observaciones
plt.figure(figsize=(10, 6))
plt.scatter(df['consumo'], df['predicciones'], alpha=0.5, label='Predicciones')
plt.plot(df['consumo'], df['consumo'], color='red', linestyle='--', lw=2, label='Línea de 45°')
plt.xlabel('Consumo observado')
plt.ylabel('Consumo predicho')
plt.title('Consumo observado vs. predicho')
plt.legend()
plt.show()

# Gráficos de Dispersión de Consumo vs. Variables Predictoras con Líneas de Regresión
plt.figure(figsize=(12, 10))
for i, col in enumerate(['precio_gas', 'precio_electricidad', 'precio_petroleo', 'precio_gas_liquido', 'grados_calefaccion', 'ingreso'], start=1):
    plt.subplot(3, 3, i)
    sns.regplot(x=col, y='consumo', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'lw': 1})
    plt.title(f'Consumo vs. {col}')
plt.suptitle('Gráficos de Regresión de Consumo vs. Variables Predictoras', y=1.02)
plt.tight_layout()
plt.show()

# Visualización de la Matriz de Correlación
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacion_predictores, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación entre Variables')
plt.show()