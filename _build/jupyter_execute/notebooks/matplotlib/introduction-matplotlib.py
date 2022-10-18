#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/javlintor/curso-python-us/blob/main/notebooks/matplotlib/introduction_to_matplotlib.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Introducción a Matplotlib 

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np

x = np.arange(1000)


# In[3]:


y = x ** 2


# In[4]:


get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

plt.plot(x, y)
plt.show()


# In[5]:


plt.plot(x, y)
plt.show()


# In[ ]:




