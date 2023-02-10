#!/usr/bin/env python
# coding: utf-8

# # Gestión de proyectos en Python 
# 
# En esta sección vamos a hacer un recopilatorio de herramientas que nos pueden ayudar a la hora de desarrollar, estandarizar y compartir un proyecto de Python. Existen una amplica gama de herramientas de este tipo, pero aquí expongo las que encuentro más útiles en mi día a día

# ## Gestión de paquetes - Poetry  
# 
# <div style="display: flex; align-items: center; justify-content: center;">
#     <img src="https://drive.google.com/uc?id=1ABuxOzaBc6WkuUb-9XLtTAHnQcvGPaSc"/>
# </div>
# 
# [Poetry](https://python-poetry.org/) es un sistema de gestión de proyectos en Python que va más allá de manejar las dependencias de un proyecto como podemos hacer con [pipenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)  
# Algunos lenguanjes como go o rust tienen utilidades de líneas de comandos integrados que nos permiten manejar detalles sobre nuestro proyecto 
# 
# - configuración e instalaciones 
# - manejo de dependencias 
# - compilación y ejecuciones 
# 
# Python no tiene una utilidad nativa pero sistemas desarrollados por terceros nos pueden ayudar a realizar estas tareas de forma precisa, como es el caso de poetry. 

# ### Instalación 
#  
# No se instala como un paquete de Python, si no como una aplicación independiente. Por ejemplo, podemos ejecutar el siguiente comando 
# 
# ```
# curl -sSL http://install.python.python-poetry.org | python3 -
# ```
# 
# aunque los detalles de instalación pueden cambiar en función de cómo tengamos configurado nuestro sistema. Para verificar que hemos instalado correctamente poetry podemos ejecutar 
# 
# ```
# poetry --version 
# ```
# 
# y obtener un output como el siguiente `Poetry (version 1.3.1)`. 

# ### Creando un nuevo proyecto 
# 
# Para crear un proyecto con poetry (podemos pensar que vamos a desarrollar un paquete de python) utilizamos el siguiente comando 
# 
# ```
# poetry new py4d
# ```
# 
# Esto debería de crear una carpeta py4d con la siguiente estructura 
# 
# - Un archivo README.md como descripción del paquete 
# - Otro archivo pyproject.toml de configuración, que no debemos tocar directamente
# - Una carpeta de tests para incluir nuestros tests unitarios
# - Otra carpeta py4d donde estarán los ficheros con las funciones y clases de nuestro paquete. 
# 
# ```
# py4d
# ├── pyproject.toml
# ├── README.md
# ├── py4d
# │   └── __init__.py
# └── tests
#     └── __init__.py
# 
# ```
# 
# Por defecto, un nuevo proyecto no tiene un entorno virtual asociado por defecto, si no que Poetry utiliza en entorno virtual que crea en 
# 
# - macOS: `~/Library/Caches/pypoetry`
# - Windows: `C:\Users\<username>\AppData\Local\pypoetry\Cache`
# - Unix: `~/.cache/pypoetry`
# 
# Podemos configurar poetry para que los entornos virtuales se creen directamente en el directorio del proyecto, para ello es necesario ejecutar el siguiente comando 
# 
# ```
# poetry config virtualenvs.in-project true
# ```
# 
# Así que en principio no tenemos que crear nada más para empezar a añadir dependencias. Sólo tenemos que modificar el `pyproject.toml`, en concreto la sección `tool.poetry.dependencies` o más fácil, añadirlas con la CLI de poetry
# 
# ```
# poetry add pandas
# ```
# 
# Poetry tiene [una sintaxis de expecificación de dependencias](https://python-poetry.org/docs/dependency-specification/) muy rica, como por ejemplo si queremos replicar justo la versión que hay en google colab de pandas,
# 
# ```
# poetry add pandas==1.3.5
# ```
# 
# O por ejemplo con `pandas=^1.3.5` estamos indicando que queremos una versión superior o igual a la `1.3.5` pero siempre menor que la `2.0.0`. 

# ### Manejando dependencias 
# 
# Otra cosa muy interesante que nos proporciona poetry es la de crear [*grupos* para las dependencias](https://python-poetry.org/docs/managing-dependencies/). Por ejemplo en un proyecto puede ser deseable tener un entorno de desarrollo en el que tengamos herramientas de formateo, testing, paquetes adicionales etc y otro más minimalista para producción. Los grupos se pueden crear en el archivo de configuración `pyproject.toml`, por ejemplo 
# 
# ```
# [tool.poetry.group.test] 
# 
# [tool.poetry.group.test.dependencies]
# pytest = "^6.0.0"
# pytest-mock = "*"
# ```
# 
# Si queremos que las dependencias de un grupo no se instalen por defecto, tenemos que añadir el parámetro `optional` en la configuración 
# 
# ```
# [tool.poetry.group.test] 
# optional = true
# ```
# 
# Podemos instalar las dependencias por defecto o de un grupo concreto utilizando `poetry install` (con las opciones `--with` o `--without`, véase la documentación). Por ejemplo si quremos actualizar solo las dependencias de un grupo, podemos hacer 
# 
# ```poetry update --with test```
# 
# Nótese que con el comando `poetry add` las dependencias se instalan automáticamente. 

# ### Usando poetry
# 
# Podemos lanzar un script con poetry simplemente escribiendo `poetry run python your_script.py`. Del mismo modo si utilizas herramientas desde línea de comandos puedes lanzarlas con poetry, como `poetry run black .`
# 
# Finalmente, existen funcionalidades más allá como la de construir paquetes distribuible y publicar que se basan en los comandos, `poetry build` y `poetry publish`, pero eso lo dejamos para otro día. 

# ## Limpia tu código - flake8 + black
# 
# ### flake8
# 
# [flake8](https://github.com/pycqa/flake8) es un linter, es decir, una herramienta que nos dará información sobre qué partes de nuestro código no siguen determinadas guías de estilo o tienen alguna lógica herrónea, como asignar variables que no se utilizan o hacer los imports en el orden inadecuado. 
# 
# flake8 nos informará a través de una serie de códigos basado en `pycodestyle` sobre qué debemos mejorar de nuestro código. En lugar de hacer los cambios a mano, puede ser interesante utilizar un formateador como black, que lo hará de forma automática y precisa por nosotros. 
# 
# ### Black 
# 
# <div style="display: flex; align-items: center; justify-content: center;">
#     <img src="https://drive.google.com/uc?id=1xe6unReOz8Av38lNMllJKytQQidKdPJq"/>
# </div>
# 
# 
# [Black](https://github.com/psf/black) es una herramienta de **formateo** de código, es decir, al aplicar black a nuestro código, va a modificar nuestro código según para que siga ciertas guías de estilo, normalmente basadas en [PEP8](https://peps.python.org/pep-0008/), que son configurables por nosotros mismos, de modo que se cambiará el estilo del código pero nunca se modificará el output del intérprete de Python. 
# 
# [Black](https://black.readthedocs.io/en/stable/) tiene un principio básico y es que todo el código que formatea debe ser similar, haciendo que estilo pueda cambiar bastante del de partida, pero minimizando los cambios respecto a las diferentes versiones del proyecto u otros proyectos igualmente formateados. 
# 
# Podemos instalar black utilizando `pip`, siempre que tengamos Python 3.7 o superior. 
# 
# ```
# pip install black
# ```
# 
# Si queremos formatear nuestros notebooks de jupyter, debemos instalarlo con 
# 
# ```
# pip install black[jupyter]
# ```
# 
# Se emplea de forma muy sencilla, simplemente llamando al comando `black` en una terminal 
# 
# ```
# black {fichero_que_queremos_formatear.py}
# ```
# 
# Tanto flake8 como black son ejemplos de linters y formatter pero hay muchos más, cada uno con sus ventajas e inconvenientes. Son altamente configurables (por ejemplo, para alterar algunas de las normas de estilo) y suelen ser muy útiles combinadas con git, la herramienta de control de versiones de la que ya hemos hablado en el curso. En particular, podemos usar herramientas como `pre-commit` para que estas herramientas se ejecuten cada vez que realizamos un commit o similar. 

# ## Logs
# 
# El **logging** es un mecanismo para registrar información sobre la ejecución de un programa, lo que puede ser muy útil para depurar errores, auditar el uso del sistema, y tomar decisiones basadas en los datos de registro. El mecanismo de logging más básico es simplemente hacer `print` por stdout, pero hay herramientas más sofisticadas que nos permiten ir más allá. Existen varios módulos de logging en Python, pero el más comúnmente utilizado es el [logging](https://docs.python.org/3/library/logging.html) que viene con la biblioteca estándar. Con este módulo, puedes especificar el nivel de detalle que deseas registrar, elegir dónde escribir tus registros (por ejemplo, en un archivo o en la consola), y ajustar el formato de tus mensajes de registro. 

# Aquí hay algunos ejemplos básicos de código para usar la biblioteca de logging en Python:

# In[ ]:


import logging

# Configuramos un registrador
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuramos un manejador para escribir en consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Configuramos un formato para los mensajes de registro
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Agregamos el manejador al registrador
logger.addHandler(console_handler)

# Usamos el registrador en nuestro código
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')


# Si cambios el nivel de logging, sólo los mensajes con un nivel más alto permanecerán

# In[ ]:


logger.setLevel(logging.WARNING)


# In[ ]:


logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')


# Si queremos sacar el log por un fichero en lugar de por consolta, deberemos añadir un handler del siguiente tipo 

# In[ ]:


file_handler = logging.FileHandler('example.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# In[ ]:


logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')


# ## Tests
# 
# Los **test unitarios** son pruebas automatizadas que se realizan para asegurarse de que una pequeña porción de código, conocida como una "unidad", funciona correctamente. Estos tests son importantes porque permiten detectar errores en el código de manera temprana, lo que facilita la depuración y aumenta la confianza en el código.
# 
# En Python, existen varias bibliotecas que se utilizan para crear test unitarios, pero una de las más comúnmente utilizadas es [unittest](https://docs.python.org/3/library/unittest.html), que viene incluida en la biblioteca estándar.
# 
# Un test unitario se compone de una o más pruebas que se realizan sobre el código, y pueden comprobar si el código devuelve el resultado esperado para diferentes entradas. Por ejemplo, si tienes una función que calcula el área de un círculo, podrías escribir un test unitario que verifique que la función devuelve el resultado correcto

# In[ ]:


import unittest
import math

def area_circle(radius):
    return math.pi * radius**2

class TestAreaCircle(unittest.TestCase):
    def test_area(self):
        self.assertIsInstance(area_circle(1), float)
        self.assertAlmostEqual(area_circle(1), math.pi)
        self.assertAlmostEqual(area_circle(2), 4 * math.pi)
        self.assertAlmostEqual(area_circle(3), 9 * math.pi)


# Hay una [amplia lista de métodos](https://docs.python.org/3/library/unittest.html#assert-methods) que pueden ser ejecutados a la hora de testear una función. Hay un método `setUp` especial que se ejecuta antes de llamar al resto de métodos.
# 
# 

# In[ ]:


class TestAreaCircle(unittest.TestCase):
    def SetUp(self):
        self.area_one = area_circle(1)

    def test_area(self):
        self.assertIsInstance(self.area_one, float)
        self.assertAlmostEqual(self.area_one, math.pi)
        self.assertAlmostEqual(area_circle(2), 4 * math.pi)
        self.assertAlmostEqual(area_circle(3), 9 * math.pi)


# ## Dash
# 
# [Dash](https://dash.plotly.com/) es un framework para hacer aplicaciones web interactivas para Python. Es una biblioteca construida en top de Flask, Plotly.js y React.js que permite crear aplicaciones web con gráficos interactivos y dashboards en tiempo real. Dash hace fácil crear interfaces de usuario complejas con una gran cantidad de elementos interactivos, incluyendo gráficos, tablas, deslizadores, botones, entre otros.
# 
# <div style="display: flex; align-items: center; justify-content: center;">
#     <img src="https://drive.google.com/uc?id=1kEdTYfWSWHhWP6WsmxlQpPAeI2lhmHXr"/>
# </div>
# 
# El siguiente código crea una aplicación y la expone en el puerto 8050 de nuestro localhost

# In[2]:


get_ipython().system('pip install dash')


# In[3]:


import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

app = dash.Dash()

df = pd.DataFrame({
    "Fruit": ["Apples", "Bananas", "Cherries"],
    "Amount": [3, 4, 5]
})

fig = px.bar(df, x="Fruit", y="Amount")

app.layout = html.Div(children=[
    html.H1(children="My Simple Dashboard"),
    dcc.Graph(
        id="example-graph",
        figure=fig
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)


# Lo interesante de dash es que podemos modificar las propiedades del layout con las funciones **callback**, de modo que podemos incluir interactividad en nuestro dashboard. Por ejemplo, en esa aplicación el usuario escribe un número y Python lo multiplica por dos, mostrando el resultado también en la aplicación 

# In[ ]:


from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value=10, type='number')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value*2}'


if __name__ == '__main__':
    app.run_server(debug=True)

