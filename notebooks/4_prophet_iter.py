#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import time

start_time = time.time()


# In[110]:


sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")

np.set_printoptions(precision=4, threshold=10000, linewidth=100, edgeitems=999, suppress=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 10)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 6)

pd.set_option('display.max_columns', 100)

# Styles
sns.set_context('poster')
sns.set_style('darkgrid')
sns.set_palette("Paired")

plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.color'] = 'k'

font = {'size'   : 10}

plt.rc('font', **font)


# In[111]:


train_store = pd.read_pickle("../data/1_train_store_preprocessed.pkl").sort_values(['Store','Date'])
test_store = pd.read_pickle("../data/1_test_store_preprocessed.pkl").sort_values(['Store','Date'])
test = pd.read_csv("../data/test.csv", sep=",", parse_dates=['Date'],
                  dtype={'StateHoliday': str, 'SchoolHoliday':str})


# In[120]:




# In[4]:


import warnings
import itertools
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm


# In[5]:


import pandas as pd
from fbprophet import Prophet


# In[66]:

forecast_total = pd.DataFrame()

NN = test['Store'].unique()

for i,N in enumerate(NN):
	print()
	print()
	print(f'{N} ({i+1}/{len(NN)})')
	train_store_N = train_store[train_store['Store']==N]
	test_store_N = test_store[test_store['Store']==N]

	#print(len(train_store_N),len(test_store_N))

	train_store_N.tail()


	# In[67]:


	df_tot = train_store_N[['Date','Sales']]
	df_tot.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
	df_tot.head()


	# In[68]:


	# Do we sell on sundays?
	df_tot[df_tot['ds'].dt.dayofweek == 6]['y'].value_counts()
	# No...


	# In[69]:


	# REMOVE SUNDAYS
	# df = df_tot
	df = df_tot[df_tot['ds'].dt.dayofweek < 6]
	df.head(10)


	# In[70]:


	holidays_1 = train_store_N[['Date','StateHoliday','SchoolHoliday','Promo']]
	holidays_1['StateHoliday'].value_counts(sort=False)


	# In[71]:


	holidays_test = test_store_N[['Date','StateHoliday','SchoolHoliday','Promo']]
	holidays_test['StateHoliday'].value_counts(sort=False)


	# In[72]:


	#holidays_1[['StateHoliday','SchoolHoliday','Promo']].value_counts(sort=False)


	# In[73]:


	holidays_tot = pd.DataFrame()


	# In[74]:


	holiday_col = 'StateHoliday' 

	list_id = holidays_1[holiday_col].unique()

	#print(holiday_col)
	for k in list_id:
		if k == 0:
			continue
		#print(k,end=' - ')
		y = holidays_1[holidays_1[holiday_col]==k]['Date']
		holidays_2 = pd.DataFrame({
		  'holiday': f'{holiday_col} - {k}',
		  'ds': pd.to_datetime(y.to_list()),
		  'lower_window': 0,
		  'upper_window': 1,
		})
		y = holidays_test[holidays_test[holiday_col]==k]['Date']
		holidays_3 = pd.DataFrame({
		  'holiday': f'{holiday_col} - {k}',
		  'ds': pd.to_datetime(y.to_list()),
		  'lower_window': 0,
		  'upper_window': 1,
		})
		#print(f'{len(holidays_2)}+{len(holidays_3)}',end='   ')
		holidays_tot = pd.concat((holidays_tot, holidays_2,holidays_3),ignore_index=True)
	holidays_tot.tail()

	len(holidays_tot),len(df)


	# In[76]:


	m = Prophet(holidays=holidays_tot, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, seasonality_mode = 'multiplicative',
			   changepoint_prior_scale=0.05)
	m.add_country_holidays(country_name='DE')
	#m2.add_regressor("my_holiday")
	m.fit(df)


	# In[77]:


	test_store['Date'].min(),test_store['Date'].max()


	# In[105]:


	future = m.make_future_dataframe(periods=6*8, freq='D')
	future.min(),future.max()


	# In[121]:


	forecast = m.predict(future)
	forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)


	# In[122]:


	mask = (forecast['ds'] >= test_store['Date'].min()) & (forecast['ds'] <= test_store['Date'].max())
	forecast_main = forecast.loc[mask]
	forecast_main['ds'].min(),forecast_main['ds'].max()


	# In[123]:


	forecast_main['Store']=N
	forecast_main.rename(columns={'ds': 'Date','yhat': 'Sales'}, inplace=True)
	forecast_main['Sales']=np.round(forecast_main['Sales'])
	forecast_main.loc[forecast_main['Date'].dt.dayofweek == 6,'Sales']=0
	forecast_main[['Store','Date','Sales']].head(10)


	# In[124]:


	forecast_total = pd.concat((forecast_total,forecast_main[['Store','Date','Sales']]))
	forecast_total.head(10)


# In[125]:


submission = pd.merge(test[['Id','Store','Date']], forecast_total,  how='left', left_on=['Store','Date'], right_on = ['Store','Date'])
submission[submission['Store']==N].head(10)

# In[ ]:
submission[['Id','Sales']].to_csv("./4_prophet.csv", sep=',', index=False)

