#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import pingouin as pg
from pyprocessmacro import Process


# In[3]:


df = pd.read_csv("Database_individual_assignment.csv")
df


# In[4]:


df.dtypes


# In[10]:


df.isna().sum(axis = 0)


# In[8]:


df.isnull().sum(axis = 0)


# In[11]:


df.replace(' ','Missing',inplace=True)


# In[12]:


df.replace('Missing', np.nan, inplace=True)
df


# In[13]:


df.isnull().sum(axis = 0)


# In[14]:


df1 = df.dropna()
df1


# In[40]:


df1 = df1.apply(pd.to_numeric, errors='coerce')


# In[41]:


df1.shape


# In[42]:


df2 = df1.iloc[:, 5:56]
df2


# In[43]:


fa = FactorAnalyzer(n_factors=7, rotation='varimax')
fa.fit(df2)
ev, v = fa.get_eigenvalues()
ev


# In[44]:


sorted(ev)


# In[45]:


plt.scatter(range(1,df2.shape[1]+1),ev)
plt.plot(range(1,df2.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[46]:


pd.DataFrame(fa.loadings_, columns=['Factor1','Factor2','Factor3','Factor4','Factor5','Factor6','Factor7'],index=[df2.columns])


# In[47]:


df3 = df2.iloc[:, 0:10]
df3


# In[48]:


pg.cronbach_alpha(data=df3)


# In[49]:


df3.dtypes


# In[81]:


df4 = df2.iloc[:, 10:20]
pg.cronbach_alpha(data=df4)


# In[80]:


df5 = df2.iloc[:, 20:30]
pg.cronbach_alpha(data=df5)


# In[82]:


df6 = df2.iloc[:, 30:34]
pg.cronbach_alpha(data=df6)


# In[83]:


df7 = df2.iloc[:, 34:37]
pg.cronbach_alpha(data=df7)


# In[84]:


df8 = df2.iloc[:, 37:42]
pg.cronbach_alpha(data=df8)


# In[85]:


df9 = df2.iloc[:, 42:56]
pg.cronbach_alpha(data=df9)


# In[86]:


df1 = df1.astype('float')


# In[87]:


df1['performance'] = df1['perfa_t0']+df1['perfb_t0']+df1['perfc_t0']+df1['perfd_t0']
df1['perfavr'] = df1['performance']/4
df1


# In[88]:


df1['efficacy'] = df1['selfa_t0']+df1['selfb_t0']+df1['selfc_t0']+df1['selfd_t0']+df1['selfe_t0']+df1['selff_t0']+df1['selfg_t0']+df1['selfh_t0']++df1['selfi_t0']
df1['effavr'] = df1['efficacy']/9
df1


# In[89]:


df1['burnout'] = df1['burna_t0']+df1['burnb_t0']+df1['burnc_t0']+df1['burnd_t0']+df1['burne_t0']+df1['burnf_t0']+df1['burng_t0']+df1['burnh_t0']+df1['burni_t0']
df1['bnoutavr'] = df1['burnout']/9
df1


# In[90]:


df10 =df1[['workexperience','age','gender','perfavr','effavr','bnoutavr']]
df10


# In[92]:


def boxplot(column):
    sns.boxplot(data=df10,x=df10[f"{column}"])
    plt.title(f"Boxplot of {column}")
    plt.show()


# In[93]:


boxplot('workexperience')
boxplot('age')
boxplot('gender')
boxplot('perfavr')
boxplot('effavr')
boxplot('bnoutavr')


# In[94]:


df11 = df10[(df10['age']<26)&(df10['perfavr']>3.5)&(df10['effavr']>2.5)&(df10['bnoutavr']<4)].copy()
df11


# In[96]:


df11.describe()


# In[97]:


df11.corr()


# In[98]:


p = Process(data=df11, model=4, x="effavr", y="perfavr", m=["bnoutavr"], controls=["workexperience", "gender"], center=True, boot=10000)
p.summary()


# In[100]:


p2 = Process(data=df11, model=1, x="effavr", y="bnoutavr", m="age", controls=["workexperience", "gender"], center=False, boot=10000)
p2.summary()


# In[101]:


p3 = Process(data=df11, model=7, x="effavr", y="perfavr", w="age", m=["bnoutavr"],controls=["workexperience","gender"],center=True, boot=10000)
p3.summary()


# In[ ]:




