#!/usr/bin/env python
# coding: utf-8

# (pandas)=
# # Pandas

# En esta sección haremos una introducción a la librería [pandas](https://pandas.pydata.org/) de Python, una herramienta muy útil para el análisis de datos. Proporciona estructuras de datos y operaciones para manipular y analizar datos de manera rápida y eficiente, así como funcionalidades de lectura y escritura de datos en diferentes formatos, como CSV, Excel, SQL, entre otros. También permite realizar operaciones matemáticas y estadísticas en los datos, así como visualizarlos en gráficos y tablas de manera cómoda. En resumen, pandas es una librería muy útil para cualquier persona que trabaje con datos y necesite realizar análisis y operaciones en ellos de manera rápida y eficiente.

# <div style="display: flex; align-items: center; justify-content: center;">
#     <img src="../../../../images/pandas.png"/>
# </div>

# La integración entre numpy y pandas se realiza mediante el uso de los arrays de numpy como el tipo de dato subyacente en las estructuras de datos de pandas. Esto permite que pandas utilice la eficiencia y la velocidad de cálculo de numpy en sus operaciones, mientras que proporciona una interfaz de usuario más amigable y especializada para trabajar con datos tabulares.
# 
# Por ejemplo, uno puede crear un DataFrame de pandas (que es una estructura de datos tabular) a partir de un arreglo de NumPy con el siguiente código:

# In[1]:


import pandas as pd
import numpy as np

# Crear un array de NumPy
arr = np.random.rand(5, 4)

# Convertir el array en un DataFrame de pandas
df = pd.DataFrame(arr)


# 
