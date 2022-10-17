#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/javlintor/curso-python-us/blob/main/Introduccion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Introducción a Python 
# 
# [Python](https://www.python.org/) es un **lenguaje de programación** altamente popular y con una sintaxis sencilla, lo que lo hace fácilmente accesible para programadores no experimentados. No obstante, es una herramienta muy potente que puede ayudarnos a resolver problemas de muy diferente índole. 
# 
# Entre sus características principales destacan: 
# - Es un lenguaje **interpretado**. 
# - Es de **tipado fuerte y dinámico**. 
# - Es **multiplataforma**. 
# - Tiene una amplia **librería estándar** además de una extensa comunidad que da soporte a una variedad de paquetes de terceros. El repositorio de referencia para paquetes de Python es el [Python Package Index](https://pypi.org/)
# 
# Podemos visitar [la encuesta anual de Stackoverflow](https://survey.stackoverflow.co/2022/#technology-most-popular-technologies) sobre uso y opinión de diferentes tecnologías para hacernos una idea de la presencia del entorno Python en el mundo del desarrollo, sobre todo en campos cercanos a la Ciencia de Datos. 

# ## Descargar Python
# 
# Podemos descargar Python desde [su sitio oficial](https://www.python.org/). Actualmente, la última versión disponible es la 3.10.8, aunque en este curso utilizaremos la versión 

# In[2]:


import sys
print(sys.version)


# Python incluye un gestor de paquetes llamado [pip](https://pypi.org/project/pip/), que nos permite instalar/desinstalar/actualizar paquetes alojados en el Python Project Index u otros repositorios. Además existen distribuciones como [anaconda](https://www.anaconda.com/) que nos facilitan el manejo de paquetes. 
# 
# En este curso no nos tendremos que preocupar de manejar paquetes y sus dependencias ya que trabajaremos con **Google Colab**, que nos proporciona directamente el acceso a un gran número de paquetes y funcionalidades del ecosistema Python.

# ## Dónde ejecutar Python
# 
# Hay muchos entornos en los que un intérprete de Python puede correr como un IDE (Integrated Development Environment -como [Visual Studio Code](https://code.visualstudio.com/), [Pycharm](https://www.jetbrains.com/es-es/pycharm/) o [Atom](https://atom.io/)-), un navegador o una terminal. 
# 
# Para ejecutar Python 3 desde la terminal, escribimos `python3` en el prompt y se lanzará una aplicación 
# 
# ```
# Python 3.9.1 (v3.9.1:1e5d33e9b9, Dec  7 2020, 12:10:52)
# [Clang 6.0 (clang-600.0.57)] on darwin
# Type "help", "copyright", "credits" or "license" for more information.
# >>>
# ```
# 
# Una herramienta muy utilizada es el entorno de trabajo [**Jupyter**](https://jupyter.org/), que puede ser instalado vía `pip` mediante 
# ```
# pip install notebook
# ```
# Jupyter Notebook lanza una aplicación que por defecto corre en el puerto 8888 de nuestro `localhost` en la que mediante un kernel de Python podemos ir ejecutando código en celdillas. Esto nos permite tener feedback casi inmedianto de nuestras implementaciones y es una herramienta adecuada cuando queramos experimentar o familiarizarnos con algún paquete. 
# 
# Google Colab nos permite trabajar con notebooks en nuestro navegador sin necesidad de instalar Python ni jupyter en nuestra máquina. En este caso perdemos capacidad para configurar nuestro entorno, pero estaremos alineados a la hora de seguir el curso. 
# 
# ## Notebooks de Jupyter/Colab
# 
# Ya sea en Jupyter o en Google Colab, trabajaremos con cuadernos (o notebooks) de Python, que son archivos de texto con la extensión `.ipynb` (a diferencia de los archivos `.py`, correspondientes a los ejecutables y módulos de Python). 
# 
# Cada cuaderno está compuesto por una secuencia de celdillas, que pueden ser de tipo *ejecutable* o *markdown*. En las cendillas de tipo ejecutable escribiremos nuestro código Python, mientras que en las celdas de tipo markdown podremos crear cabeceras, listas, resaltar texto con **negrita** o *cursiva* e incluso incorporar fórmulas matemáticas con *Latex*
# 
# $$
# f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
# $$
# 
# Todo ello para explicar y realizar anotaciones sobre nuestro código. 

# ## Más sobre Python 
# 
# Python fue creado en 1991 por Rossum y su nombre se debe a un famoso grupo de comedia británico llamado *Monty Python*, de ahí que en muchos ejemplos encontremos variables dummy llamadas como `spam` y `eggs` en lugar de los típicos `foo`, `bar`. 
# 
# Su popularidad se debe principalmente a que es fácil de aprender y es utilizado para construir aplicaciones en el lado del servidor a través del framework `django`, así como el tratamiento y análisis de datos en ecosistemas **big data** y el desarrollo de soluciones basadas en el **machine learning**.
# 
# Python es un lenguaje que hace especial énfasis en su **legibilidad**, evitando el uso de corchetes y semicolons en favor de la indentación. La filosofía de un buen *pythonista* puede leerse en *The Zen of Python* 

# In[1]:


import this


# In[ ]:




