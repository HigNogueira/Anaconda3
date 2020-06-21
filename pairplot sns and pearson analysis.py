#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np

from pandas import Series, DataFrame


# In[21]:


dados = np.arange(6)


# In[22]:


dados


# In[23]:


np.random.seed(25)
indice = ['linha 1', 'linha2','linha 3', 'linha 4','linha 5', 'linha 6']
colunas = [ 'coluna 1', 'coluna 2', 'coluna 3', 'coluna 4','coluna 5', 'coluna 6']

df = DataFrame(np.random.rand(36).reshape((6,6)),
                index=indice,
                columns=colunas)
df
#elaboração de uma planilha aleatoria utilizando metodo random


# In[24]:


df <.2


# In[25]:


df >.5


# In[45]:


df.plot()


# In[31]:


#para criar um filtro especifico 

indice = ['linha 1', 'linha 2',]
series_obj = Series(np.arange(2), index = indice)


# In[32]:


filtro = series_obj
series_obj [filtro]


# In[34]:


series_obj ['linha 1'] = 8
series_obj
#atualizando dados


# In[35]:


a = np.array([1,2,3,4,5,6,])
a


# In[36]:


import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib import rcParams

import seaborn as sns


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 6,5
sns.set_style('whitegrid')


# In[47]:


x = range(1,13)
y = [1,2,3,4,5,6,0,5,4,3,2,1]

plt.plot(x,y)
plt.show()
#grafico atraves do matplotlib


# In[53]:


plt.(kind='bar')
#só funciona com planilha improtada que possua uma ou mais colunas.


# In[61]:


#grafico de linha e de barra pelo matplotlib
cor = ['salmon']
plt.bar(x,y,color=cor)
plt.show()


# In[55]:


plt.savefig('grafico de barra eixo x,y')


# In[78]:


x = ['grafico de barra eixo x,y']
sns.pairplot(x)


# In[56]:


get_ipython().system(' dir')


# In[65]:


x1 = range (1,13)
y1 = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(x,y)
plt.show()


# In[66]:


x1 = range (1,13)
y1 = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(x1,y1)
plt.show()


# In[67]:


x1 = range (1,13)
y1 = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(x,y)
plt.plot(x1,y1)
plt.show()


# In[69]:


plt.savefig


# In[71]:


np.random.seed(25)
indice = ['linha 1', 'linha2','linha 3', 'linha 4','linha 5', 'linha 6', 'linha 7', 'linha 8', 'linha 9', 'linha 10']
colunas = [ 'coluna 1', 'coluna 2', 'coluna 3', 'coluna 4','coluna 5', 'coluna 6', 'coluna 7', 'coluna 8', 'coluna 9', 'coluna 10']

df = DataFrame(np.random.rand(100).reshape((10,10)),
                index=indice,
                columns=colunas)
df


# In[73]:


df.plot()
plt.show()


# In[ ]:





# In[80]:


plt.savefig ('x')


# In[75]:


from scipy.stats.stats import pearsonr


# In[81]:


sns.pairplot(x)


# In[79]:


caminho = '‪C:\Users\Igor\Downloads\mtcars.csv'
carros = pd.read_csv(caminho)


# In[ ]:




