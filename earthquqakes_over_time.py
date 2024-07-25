#import librarires
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px

#defining constants
years_per_plot = 64
binning_time = 4 #in weeks

#import datasets
etas = pd.read_csv('datasets\\ETAS.csv', sep=',', lineterminator='\n')
usgs = pd.read_csv('datasets\\USGS.csv', sep=',', lineterminator='\n')

#magnitude filtering
etas = etas[etas['mag'] > 3]
usgs = usgs[usgs['mag'] > 3]
etas['date'] = pd.to_datetime(etas['date'])

#some cleaning
etas['aftershock'] = etas['aftershock\r']
etas = etas.drop(columns='aftershock\r')
etas['aftershock'] = etas['aftershock'].str.replace('\r', '')

# Determine the start and end dates of the dataset
start_date = etas['date'].min()
end_date = etas['date'].max()

#binning data based on binning time
def binning_data(df: pd.DataFrame, binning_time:str, start_date):
    weekno = [int(i/np.timedelta64(binning_time, 'W')) for i in pd.to_datetime(df['date']) - pd.to_datetime(start_date)]
    df['weekno'] = weekno
    return df

start_date = etas['date'][0]
grouped_etas = binning_data(etas, binning_time, start_date)
print(grouped_etas)

#plotting etas earthquakes
while start_date < end_date:
    end_chunk_date = start_date + pd.DateOffset(years=years_per_plot)
    etas_chunk = grouped_etas[(grouped_etas['date'] >= start_date) & (grouped_etas['date'] < end_chunk_date)]

    longitude = etas_chunk['longitude'].to_numpy()
    latitude = etas_chunk['latitude'].to_numpy()
    mag = etas_chunk['mag'].to_numpy()
    aftershock = etas_chunk['aftershock'].to_numpy()

    date = etas_chunk['date'].to_numpy()
    weekno = etas_chunk['weekno'].to_numpy()
    
    fig = px.scatter(
        x=longitude,
        y=latitude,
        color=mag,
        size=mag,
        symbol=aftershock,
        animation_frame=weekno,
        title=f'Earthquakes Locations (ETAS) - {start_date.year}-{end_chunk_date.year}',
        color_continuous_scale='Viridis',
        opacity=0.7,
        size_max=10,
        width=800,
        height=800
    )
    
    fig.update_traces(showlegend=False)
    fig.show()

    start_date = end_chunk_date