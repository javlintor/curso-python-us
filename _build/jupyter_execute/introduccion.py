#!/usr/bin/env python
# coding: utf-8

# # Introducción a Python
# 
# A lo largo del curso vamos a desarrollar código en Jupyter notebook. Puedes ejecutar cada uno de los cuadernos en el botón de `launch > Colab`. 
# 
# Recuerda que una matrix es unitaria si 
# 
# $$ 
# A^{-1} = A^{t}
# $$ (def-unitaria)
# 
# La definición {eq}`def-unitaria` viene mejor explicada en {cite}`perez2011python`, o también puedes consultar el apartado [de la anterior lección](intro-mas-detalles)

# In[1]:


from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


# In[2]:


# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);


# There is a lot more that you can do with outputs (such as including interactive outputs)
# with your book. For more information about this, see [the Jupyter Book documentation](https://jupyterbook.org)

# In[3]:


import pandas as pd
import numpy as np

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
df.head()


# In[4]:


df.shape


# ## Añadiendo algún bloque
# 
# Como documentos markdown, puedes añadir bloques como una nota
# 
# :::{note}
# This is a note 
# :::
# 
# :::{admonition} **Ejercicio**
# Crea una clase "animal" y clases hijas para describir perros y gatos
# :::

# ### Ejercicio 4
# > *Crea una clase "animal" y clases hijas para describir perros y gatos.*
# 
# :::{admonition} Solución
# :class: dropdown
# 
# Primero vamos a definir la clase abstracta
# 
# :::{code}
# class Animal:
#     def __init__(self):
#         pass
#     def talk(self):
#         pass
# :::
# 
# :::

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 20
θ = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
colors = plt.cm.viridis(radii / 10.)

ax = plt.subplot(111, projection='polar')
ax.bar(θ, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

plt.show()


# |    Training   |   Validation   |   Other Col   |
# | :------------ | :-------------: | :-------------: |
# |        0      |        5       |        4       |
# |     13720     |      2744      |      $x = y + z$      |
