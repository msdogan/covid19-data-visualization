__author__ = 'mustafa_dogan'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, os, sys
from matplotlib import animation
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

source = 'https://github.com/CSSEGISandData/COVID-19'

# countries (and UID from UID_ISO_FIPS_LookUp_Table.csv) to extract information
countries = {
				'US':840,
				'Turkey':792,
				'Italy':380,
				'Germany':276,
				'France':250,
				'United Kingdom':826,
				'Spain':724,
				'China':156,
				'Iran':364,
				'Singapore':702,
				'Switzerland':756,
				'Australia':36,
				'Norway':578,
				'Finland':246,
				'Sweden':752,
				'Canada':124,
				'Japan':392,
				'Denmark':208,
				'Netherlands':528,
				'Korea, South':410,
				'Portugal':620,
				'Belgium':56,
				}

# confirmed cases
path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
confirmed_df = pd.read_csv(path)
confirmed = pd.DataFrame(confirmed_df.T.ix[4:].values,columns=confirmed_df.T.ix[1],index=confirmed_df.T.index[4:])
df2 = confirmed.T.groupby(['Country/Region']).sum().T
confirmed_all_df = df2.copy()
confirmed_df = df2[countries.keys()]
confirmed_df.index = pd.to_datetime(confirmed_df.index)
confirmed_all_df.index = pd.to_datetime(confirmed_all_df.index)
# print(confirmed.keys().values) # print countries

# new confirmed cases
confirmed_new_df = confirmed_df.copy()
for i in range(len(confirmed_new_df.index)-1):
	confirmed_new_df.ix[-(i+1)]=confirmed_new_df.ix[-(i+1)]-confirmed_new_df.ix[-(i+2)]

# fatalities
path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
death_df = pd.read_csv(path)
death = pd.DataFrame(death_df.T.ix[4:].values,columns=death_df.T.ix[1],index=death_df.T.index[4:])
df2 = death.T.groupby(['Country/Region']).sum().T
death_all_df = df2.copy()
death_df = df2[countries.keys()]
death_df.index = pd.to_datetime(death_df.index)
death_all_df.index = pd.to_datetime(death_all_df.index)

# new deaths
death_new_df = death_df.copy()
for i in range(len(confirmed_new_df.index)-1):
	death_new_df.ix[-(i+1)]=death_new_df.ix[-(i+1)]-death_new_df.ix[-(i+2)]

# death rate
data = death_df.ix[-1]/confirmed_df.ix[-1]*100
sorted_idx_p = np.argsort(data)
barPos_p = np.arange(sorted_idx_p.shape[0])
sc = np.array(countries.keys())

# recovered
path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recovered_df = pd.read_csv(path)
recovered = pd.DataFrame(recovered_df.T.ix[4:].values,columns=recovered_df.T.ix[1],index=recovered_df.T.index[4:])
df2 = recovered.T.groupby(['Country/Region']).sum().T
recovered_all_df = df2.copy()
recovered_df = df2[countries.keys()]
recovered_df.index = pd.to_datetime(recovered_df.index)
recovered_all_df.index = pd.to_datetime(recovered_all_df.index)

# population and lat long
path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
country_data_df = pd.read_csv(path)
country_data_df = country_data_df.set_index(['UID'])
population = country_data_df['Population'].loc[countries.values()]

# active cases
active_df = confirmed_df-death_df-recovered_df
active_all_df = confirmed_all_df-death_all_df-recovered_all_df

color = sns.color_palette("dark", len(countries.keys()))

# all in one plot 1
fig = plt.figure(figsize=(12, 8))

# confirmed trend
ax1 = plt.subplot(231)
for k,item in enumerate(confirmed_df.keys()):
	df_plot = confirmed_df[confirmed_df[item] >= 10]
	plt.plot(np.arange(0,len(df_plot[item])),df_plot[item],color=color[k],alpha=0.5,linewidth=1,label=item)
	plt.text(len(df_plot[item])-1,df_plot[item].iloc[-1],item,fontsize=8,color=color[k],alpha=0.6,fontweight='bold',verticalalignment='center')
ax1.set_yscale('log')
plt.title('Total Confirmed COVID-19 Cases', loc='left', fontweight='bold',fontsize=11)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.xlabel('Days since confirmed cases reached 10',fontsize=10)
plt.ylim([10,10**6])
plt.xlim([0,90])

ax2 = plt.subplot(232)
for k,item in enumerate(confirmed_df.keys()):
	df_plot = confirmed_df[confirmed_df[item] >= 100]
	plt.plot(np.arange(0,len(df_plot[item])),df_plot[item],color=color[k],alpha=0.5,linewidth=1,label=item)
	plt.text(len(df_plot[item])-1,df_plot[item].iloc[-1],item,fontsize=8,color=color[k],alpha=0.6,fontweight='bold',verticalalignment='center')
ax2.set_yscale('log')
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
plt.xlabel('Days since confirmed cases reached 100',fontsize=10)
plt.ylim([10**2,10**6])
plt.xlim([0,90])

ax3 = plt.subplot(233)
for k,item in enumerate(confirmed_df.keys()):
	df_plot = confirmed_df[confirmed_df[item] >= 1000]
	plt.plot(np.arange(0,len(df_plot[item])),df_plot[item],color=color[k],alpha=0.5,linewidth=1,label=item)
	plt.text(len(df_plot[item])-1,df_plot[item].iloc[-1],item,fontsize=8,color=color[k],alpha=0.6,fontweight='bold',verticalalignment='center')
ax3.set_yscale('log')
ax3.get_xaxis().tick_bottom()
ax3.get_yaxis().tick_left()
plt.xlabel('Days since confirmed cases reached 1000',fontsize=10)
plt.ylim([10**3,10**6])
plt.xlim([0,90])

# active trend
ax4 = plt.subplot(235)
active_all_df.plot(ax=ax4,alpha=0.3,color='gray',legend=False,linewidth=0.5)
active_df.plot(
	logy=True,
	ax=ax4,alpha=0.5,ylim=[100,10**6],xlim=[active_df.index[0],active_df.index[-1]],color=color,legend=False,linewidth=1)
for i,item in enumerate(active_df.keys()):
	plt.text(active_df.index[-1],active_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9,verticalalignment='center')
plt.title('Total Active COVID-19 Cases', loc='left', fontweight='bold',fontsize=11)
ax4.get_xaxis().tick_bottom()
ax4.get_yaxis().tick_left()

# death trend
ax5 = plt.subplot(236)
for k,item in enumerate(death_df.keys()):
	df_plot = death_df[death_df[item] >= 1]
	plt.plot(np.arange(0,len(df_plot[item])),df_plot[item],color=color[k],alpha=0.5,linewidth=1,label=item)
	plt.text(len(df_plot[item])-1,df_plot[item].iloc[-1],item,fontsize=8,color=color[k],alpha=0.6,fontweight='bold',verticalalignment='center')
ax5.set_yscale('log')
plt.title('Total Fatalities', loc='left', fontweight='bold',fontsize=11)
ax5.get_xaxis().tick_bottom()
ax5.get_yaxis().tick_left()
plt.xlabel('Days since the first fatality',fontsize=10)
plt.ylim([1,10**5])
plt.xlim([0,90])
plt.legend(bbox_to_anchor=(1.2, 1.2, 1., .12), loc='center right', ncol=1)

# incidence ratio
ir = []
for i,item in enumerate(countries.keys()):
	ir.append(confirmed_df[item].iloc[-1]/population[countries[item]]*10**6)
ir = np.array(ir)
sorted_idx_p_ir = np.argsort(ir)
barPos_p_ir = np.arange(sorted_idx_p_ir.shape[0])
sc_ir = np.array(countries.keys())

# incidence rate
ax6 = plt.subplot(234)
plt.barh(barPos_p_ir, ir[sorted_idx_p_ir],alpha=0.9)
plt.yticks(barPos_p_ir, sc_ir[sorted_idx_p_ir],fontsize=8)
plt.title('Incidence Rate', loc='left', fontweight='bold',fontsize=11)
ax6.get_xaxis().tick_bottom()
ax6.get_yaxis().tick_left()
plt.xlim([0,4000])
plt.ylim([-1,len(barPos_p_ir)])
plt.xlabel('Confirmed Case per Million People',fontsize=10)
for i,item in enumerate(ir[sorted_idx_p_ir]):
	plt.text(item,i,str(int(round(item))),verticalalignment='center',fontsize=7,color='gray')

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.8, top=0.925, wspace=0.6, hspace=0.4)
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=12,color='grey')
plt.gcf().text(0.35, 0.01, 'Data Source: Johns Hopkins University, '+source,fontsize=12,color='grey')
plt.savefig('confirmed_trend.png',transparent=False,dpi=400)
plt.close(fig)

# confirmed trend
fig = plt.figure(figsize=(5,4))
ax = fig.gca()

def animate(i):
	ax.clear()
	# print(i)
	item = confirmed_df.keys()[i]
	df_plot_italy = confirmed_df[confirmed_df['Italy'] >= 100]
	plt.fill_between(np.arange(0,len(df_plot_italy['Italy'])),df_plot_italy['Italy'],color='blue',alpha=0.2,linewidth=2,label='Italy')
	df_plot = confirmed_df[confirmed_df[item] >= 100]
	plt.fill_between(np.arange(0,len(df_plot[item])),df_plot[item],color=color[i],alpha=0.4,linewidth=2,label=item)
	plt.text(len(df_plot[item])/2,300,item,fontsize=10,color=color[i],alpha=0.9,fontweight='bold')
	ax.set_yscale('log')
	plt.title('Total Confirmed COVID-19 Cases', loc='left', fontweight='bold',fontsize=18)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.xlabel('Days since Confirmed Cases Reached 100',fontsize=12)
	plt.ylim([10**2,10**6])
	plt.legend(loc=2)
	# plt.xlim([0,len(df_plot_italy['Italy'])])
	plt.xlim([0,None])
	plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=8,color='grey')
	plt.gcf().text(0.3, 0.01, 'Data Source: JHU, '+source,fontsize=8,color='grey')
	plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.925)
	return None

anim = animation.FuncAnimation(fig, animate, frames=len(confirmed_df.keys()), interval=1200)
anim.save('confirmed.gif', writer='imagemagick', dpi=400)

# all in one plot 2
fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(231)
confirmed_new_df.plot(
	# logy=True,
	ax=ax1,alpha=0.5,ylim=[1,35000],xlim=[confirmed_new_df.index[0],confirmed_new_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Daily Confirmed Cases', loc='left', fontweight='bold',fontsize=11)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
for i,item in enumerate(confirmed_new_df.keys()):
	plt.text(confirmed_new_df.index[-1],confirmed_new_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9,verticalalignment='center')

ax2 = plt.subplot(232)
confirmed_all_df.plot(ax=ax2,alpha=0.3,color='gray',legend=False,linewidth=0.5)
confirmed_df.plot(
	logy=True,
	ax=ax2,alpha=0.5,ylim=[1,10**6],xlim=[confirmed_df.index[0],confirmed_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Total Confirmed Cases', loc='left', fontweight='bold',fontsize=11)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
for i,item in enumerate(confirmed_df.keys()):
	plt.text(confirmed_df.index[-1],confirmed_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9,verticalalignment='center')

ax3 = plt.subplot(233)
death_new_df.plot(
	# logy=True,
	ax=ax3,alpha=0.5,ylim=[1,2500],xlim=[death_new_df.index[0],death_new_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Daily Fatalities', loc='left', fontweight='bold',fontsize=11)
ax3.get_xaxis().tick_bottom()
ax3.get_yaxis().tick_left()
for i,item in enumerate(death_new_df.keys()):
	plt.text(death_new_df.index[-1],death_new_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9,verticalalignment='center')

ax4 = plt.subplot(234)
plt.barh(barPos_p, data[sorted_idx_p],alpha=0.9)
plt.yticks(barPos_p, sc[sorted_idx_p],fontsize=8)
plt.title('Mortality Rate (%)', loc='left', fontweight='bold',fontsize=11)
ax4.get_xaxis().tick_bottom()
ax4.get_yaxis().tick_left()
plt.xlim([0,15])
plt.ylim([-1,len(barPos_p)])
plt.xlabel('(# of Deaths / Confirmed Cases) * 100')
for i,item in enumerate(data[sorted_idx_p]):
	plt.text(item,i,str(round(item,1)),verticalalignment='center',fontsize=7,color='gray')

ax5 = plt.subplot(235)
death_all_df.plot(ax=ax5,alpha=0.3,color='gray',legend=False,linewidth=0.5)
death_df.plot(
	logy=True,
	ax=ax5,alpha=0.5,ylim=[1,10**5],xlim=[death_df.index[0],death_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Total Number of Deaths', loc='left', fontweight='bold',fontsize=11)
ax5.get_xaxis().tick_bottom()
ax5.get_yaxis().tick_left()
for i,item in enumerate(death_df.keys()):
	plt.text(death_df.index[-1],death_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9,verticalalignment='center')

ax6 = plt.subplot(236)
recovered_df.plot(
	logy=True,
	ax=ax6,alpha=0.5,ylim=[1,10**5],xlim=[recovered_df.index[0],confirmed_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Total Recovered Patients', loc='left', fontweight='bold',fontsize=11)
ax6.get_xaxis().tick_bottom()
ax6.get_yaxis().tick_left()
for i,item in enumerate(recovered_df.keys()):
	plt.text(recovered_df.index[-1],recovered_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9,verticalalignment='center')

plt.legend(bbox_to_anchor=(1.2, 1.2, 1., .12), loc='center right', ncol=1)
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.8, top=0.925, wspace=0.6, hspace=0.4)
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=12,color='grey')
plt.gcf().text(0.35, 0.01, 'Data Source: Johns Hopkins University, '+source,fontsize=12,color='grey')
plt.savefig('total_plot.png',transparent=False,dpi=400)
plt.close(fig)

