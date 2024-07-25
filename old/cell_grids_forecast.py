#import librarires
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras import layers

#defining constants
binning_time = '4W'
lat_grid_size = 0.5
lon_grid_size = 0.5

#import datasets
etas = pd.read_csv('datasets\\ETAS.csv', sep=',', lineterminator='\n')
usgs = pd.read_csv('datasets\\USGS.csv', sep=',', lineterminator='\n')

#magnitude filtering
etas = etas[etas['mag'] > 3]
usgs = usgs[usgs['mag'] > 3]
etas['date'] = pd.to_datetime(etas['date'])

# Determine the start and end dates of the dataset
start_date = etas['date'].min()
end_date = etas['date'].max()

def cell_grids(row):    
    lat_cell = int(row['latitude'] // lat_grid_size)
    lon_cell = int(row['longitude'] // lon_grid_size)
    return lat_cell, lon_cell


def grouping(df:pd.DataFrame):
    result = df.groupby(['cell_no', pd.Grouper(key='date', freq=binning_time)])['mag'].sum().reset_index()
    return result

etas['cell_no'] = etas.apply(cell_grids, axis=1)
result = grouping(etas)

min_lat, max_lat = etas['latitude'].min(), etas['latitude'].max()
min_lon, max_lon = etas['longitude'].min(), etas['longitude'].max()

num_lat_cells = int(np.ceil((max_lat - min_lat) / lat_grid_size))
num_lon_cells = int(np.ceil((max_lon - min_lon) / lon_grid_size))
zmax = result['mag'].max()

# Preallocate arrays
data = [None] * len(result['date'].unique())

heatmap = np.zeros((num_lat_cells, num_lon_cells), dtype=float)
# print(heatmap)
scaler = MinMaxScaler()

for frame, (date, frame_data) in enumerate(result.groupby('date')):
    scaled_mag = scaler.fit_transform(frame_data[['mag']])
    for _, row in frame_data.iterrows():
        lat_cell, lon_cell = row['cell_no']
        lat_idx = int(lat_cell - min_lat / lat_grid_size)
        lon_idx = int(lon_cell - min_lon / lon_grid_size)
        heatmap[lat_idx, lon_idx] = row['mag']
        
        data[frame] = [
            np.where(heatmap == 0, 0, heatmap),
        ]
    
    heatmap.fill(0)

df = pd.DataFrame(data)
# print(df)
df.to_csv('heatmap.csv')