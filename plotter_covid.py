__author__ = 'mustafa_dogan'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, os, sys
from matplotlib import animation
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

source = 'https://github.com/CSSEGISandData/COVID-19'

# countries to extract information
countries = [
				'US',
				'Turkey',
				'Italy',
				'Germany',
				'France',
				'United Kingdom',
				'Spain',
				'China',
				'Iran',
				'Singapore',
				'Switzerland',
				# 'Australia',
				'Norway',
				'Finland',
				'Sweden',
				'Canada',
				'Japan',
				'Denmark',
				'Netherlands',
				'Korea, South',
				]

path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
confirmed_df = pd.read_csv(path)
confirmed = pd.DataFrame(confirmed_df.T.ix[4:].values,columns=confirmed_df.T.ix[1],index=confirmed_df.T.index[4:])
df2 = confirmed.T.groupby(['Country/Region']).sum().T
confirmed_df = df2[countries]
confirmed_df.index = pd.to_datetime(confirmed_df.index)
path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
death_df = pd.read_csv(path)
death = pd.DataFrame(death_df.T.ix[4:].values,columns=death_df.T.ix[1],index=death_df.T.index[4:])
df2 = death.T.groupby(['Country/Region']).sum().T
death_df = df2[countries]
death_df.index = pd.to_datetime(death_df.index)

path = '/Users/msdogan/Documents/github/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recovered_df = pd.read_csv(path)
recovered = pd.DataFrame(recovered_df.T.ix[4:].values,columns=recovered_df.T.ix[1],index=recovered_df.T.index[4:])
df2 = recovered.T.groupby(['Country/Region']).sum().T
recovered_df = df2[countries]
recovered_df.index = pd.to_datetime(recovered_df.index)

color = sns.color_palette("dark", len(countries))

# confirmed trend
fig = plt.figure(); ax = plt.gca()
for k,item in enumerate(confirmed_df.keys()):
	df_plot = confirmed_df[confirmed_df[item] >= 100]
	plt.plot(np.arange(0,len(df_plot[item])),df_plot[item],color=color[k],alpha=0.5,linewidth=1,label=item)
	plt.text(len(df_plot[item])-1,df_plot[item].iloc[-1],item,fontsize=10,color=color[k],alpha=0.9,fontweight='bold')
ax.set_yscale('log')
plt.title('Total Confirmed COVID-19 Cases', loc='left', fontweight='bold',fontsize=18)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xlabel('Days since Confirmed Cases Reached 100',fontsize=12)
plt.ylim([100,200000])
plt.xlim([0,70])
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=10,color='grey')
plt.gcf().text(0.375, 0.01, 'Data Source: '+source, fontsize=10,color='grey')
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.925)
plt.savefig('confirmed_trend.png',transparent=False,dpi=400)
plt.close(fig)

# confirmed trend
fig = plt.figure(figsize=(6,5))
ax = fig.gca()

def animate(i):
	ax.clear()
	# print(i)
	item = confirmed_df.keys()[i]
	df_plot_italy = confirmed_df[confirmed_df['Italy'] >= 100]
	plt.fill_between(np.arange(0,len(df_plot_italy['Italy'])),df_plot_italy['Italy'],color='blue',alpha=0.2,linewidth=2,label='Italy')
	df_plot = confirmed_df[confirmed_df[item] >= 100]
	plt.plot(np.arange(0,len(df_plot[item])),df_plot[item],color=color[i],alpha=0.5,linewidth=1,label=item)
	plt.text(len(df_plot[item])-1,df_plot[item].iloc[-1],item,fontsize=10,color=color[i],alpha=0.9,fontweight='bold')
	ax.set_yscale('log')
	plt.title('Total Confirmed COVID-19 Cases', loc='left', fontweight='bold',fontsize=18)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.xlabel('Days since Confirmed Cases Reached 100',fontsize=12)
	plt.ylim([100,200000])
	plt.legend(loc=2)
	plt.xlim([0,len(df_plot_italy['Italy'])])
	plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=10,color='grey')
	plt.gcf().text(0.35, 0.01, 'Data Source: '+source, fontsize=10,color='grey')
	plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.925)
	return None

anim = animation.FuncAnimation(fig, animate, frames=len(confirmed_df.keys()), interval=1200)
anim.save('confirmed.gif', writer='imagemagick')

# confirmed cases plot
fig = plt.figure(); ax = plt.gca()
confirmed_df.plot(
	logy=True,
	ax=ax,alpha=0.5,ylim=[1,200000],color=color,legend=False,linewidth=1)
plt.title('Total Confirmed COVID-19 Cases', loc='left', fontweight='bold',fontsize=18)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for i,item in enumerate(confirmed_df.keys()):
	plt.text(confirmed_df.index[-1],confirmed_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9)
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=10,color='grey')
plt.gcf().text(0.375, 0.01, 'Data Source: '+source,fontsize=10,color='grey')
plt.subplots_adjust(left=0.125, bottom=0.175, right=0.85, top=0.925)
plt.savefig('confirmed.png',transparent=False,dpi=400)
plt.close(fig)

# number of deaths plot
fig = plt.figure(); ax = plt.gca()
death_df.plot(
	logy=True,
	ax=ax,alpha=0.5,ylim=[1,13000],xlim=[death_df.index[0],death_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Total Number of COVID-19 Deaths', loc='left', fontweight='bold',fontsize=18)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for i,item in enumerate(death_df.keys()):
	plt.text(death_df.index[-1],death_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9)
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=10,color='grey')
plt.gcf().text(0.375, 0.01, 'Data Source: '+source, fontsize=10,color='grey')
plt.subplots_adjust(left=0.125, bottom=0.175, right=0.85, top=0.925)
plt.savefig('death.png',transparent=False,dpi=400)
plt.close(fig)

# recovered plot
fig = plt.figure(); ax = plt.gca()
recovered_df.plot(
	logy=True,
	ax=ax,alpha=0.5,ylim=[1,100000],xlim=[recovered_df.index[0],confirmed_df.index[-1]],color=color,legend=False,linewidth=1)
plt.title('Total Recovered COVID-19 Patients', loc='left', fontweight='bold',fontsize=18)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for i,item in enumerate(recovered_df.keys()):
	plt.text(recovered_df.index[-1],recovered_df[item].iloc[-1],item,fontsize=8,color=color[i],alpha=0.9)
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=10,color='grey')
plt.gcf().text(0.375, 0.01, 'Data Source: '+source,fontsize=10,color='grey')
plt.subplots_adjust(left=0.125, bottom=0.175, right=0.85, top=0.925)
plt.savefig('recovered.png',transparent=False,dpi=400)
plt.close(fig)

# death rate
data = death_df.ix[-1]/confirmed_df.ix[-1]*100
sorted_idx_p = np.argsort(data)
barPos_p = np.arange(sorted_idx_p.shape[0])
sc = np.array(countries)

fig = plt.figure(); ax = plt.gca()
plt.barh(barPos_p, data[sorted_idx_p],alpha=0.9)
plt.yticks(barPos_p, sc[sorted_idx_p],fontsize='9')
plt.title('COVID-19 Mortality Rate (%)', loc='left', fontweight='bold',fontsize=18)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.gcf().text(0.01, 0.01, 'Last Updated: '+str(datetime.date.today()),fontsize=10,color='grey')
plt.gcf().text(0.375, 0.01, 'Data Source: '+source, fontsize=10,color='grey')
plt.xlim([0,12])
plt.xlabel('(# of Deaths / Confirmed Cases) * 100')
plt.subplots_adjust(left=0.175, bottom=0.15, right=0.925, top=0.925)
plt.savefig('death_rate.png',transparent=False,dpi=400)
plt.close(fig)
