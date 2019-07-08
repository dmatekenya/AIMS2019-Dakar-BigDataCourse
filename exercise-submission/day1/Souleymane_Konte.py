#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
from textblob import TextBlob


# In[ ]:


def mapcount(filename):
    filename = "C:/Users/DELL/Documents/BigData/day2/data/country_codes_africa.csv"
    with open(filename, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
    return lines


# In[6]:


def concatenate_strings(first_name, last_name):
    first_name = "Souleymane"
    last_name = "Konte"
    result = first_name+ ' '+last_name
    print (result)


# In[ ]:





# In[ ]:





# In[ ]:




