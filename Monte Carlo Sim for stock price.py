#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np 
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import pandas_datareader.data as pdr
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


#extract data from yahoo finance using panda data reader and store in panda data frame
yf.pdr_override()
data = pdr.get_data_yahoo('AAPL', start = '2010-01-01', end='2020-07-30')['Adj Close']


# In[78]:


#get historical log returns
log_returns = np.log(1 + data.pct_change())
log_returns.tail()
type(log_returns)


# In[79]:


#plot adjusted close
data.plot(figsize=(10,6))


# In[80]:


#plot log returns
log_returns.plot(figsize = (10,6))


# In[81]:


# mean for brownian motion
u1 = log_returns.mean()
u = pd.Series(u1)
u


# In[82]:


#variance for brownian motion
var1 = log_returns.var()
var = pd.Series(var1)
var


# In[83]:


#drift for brownian motion
drift = u - (0.5 * var)
type(drift)


# In[84]:


#standard deviation of log returns
stdev1 = log_returns.std()
stdev = pd.Series(stdev1)
stdev


# In[85]:


#make sure drift and stdev are pd series
print(type(drift), type(stdev))


# In[86]:


#change drift in to numpy array use either np.array() or XXXXXX.values
np.array(drift)
drift.values


# In[87]:


#change stdev into numpy array
stdev.values


# In[88]:


#95% chance of event occuring in norm dist stdev
norm.ppf(0.95)


# In[89]:


#numpy random number generator
x = np.random.rand(10,2)
x


# In[90]:


#input random numbers into norm dist function
norm.ppf(x)


# In[91]:


#combine 53 56 57 into one function for reduced code
Z = norm.ppf(np.random.rand(10,2))
Z


# In[92]:


#t_intervals is the number of days forecasted
#iterations is the number of stock price predictions
t_intervals = 365
iterations = 10


# In[93]:


#brownian motion equation
daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
daily_returns


# In[94]:


#set first price in prediction to last price of data
S0 = data.iloc[-1]
S0


# In[95]:


#create an empty array with same dimensions as daily_returns
price_list = np.zeros_like(daily_returns)
price_list


# In[96]:


#set first row in price_list to S0
price_list[0]
price_list[0] = S0
price_list


# In[97]:


#create a for loop to fill in numpy array using brownian motion eq
for t in range(1, t_intervals):
    price_list[t] = price_list[t-1] * daily_returns[t]


# In[98]:


#verify price_list is filled
price_list


# In[99]:


plt.figure(figsize=(10,6))
plt.plot(price_list);


# In[ ]:




