import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash

#defining constants for ML model
binning_time_ml = '4W'
lat_grid_size = 0.5
lon_grid_size = 0.5
usgs = pd.read_csv('datasets\\USGS.csv', sep=',', lineterminator='\n')
#magnitude filtering
usgs = usgs[usgs['mag'] > 3]
usgs['date'] = pd.to_datetime(usgs['date'])

#determine the start and end dates of the dataset
start_date = usgs['date'].min()
end_date = usgs['date'].max()

def cell_grids(row) -> tuple:    
    lat_cell = int(row['latitude'] // lat_grid_size)
    lon_cell = int(row['longitude'] // lon_grid_size)
    return lat_cell, lon_cell


def grouping(df:pd.DataFrame) -> pd.DataFrame:
    result = df.groupby(['cell_no', pd.Grouper(key='date', freq=binning_time)])['mag'].sum().reset_index()
    return result

usgs['cell_no'] = usgs.apply(cell_grids, axis=1)
result = grouping(usgs)

min_lat, max_lat = usgs['latitude'].min(), usgs['latitude'].max()
min_lon, max_lon = usgs['longitude'].min(), usgs['longitude'].max()

num_lat_cells = int(np.ceil((max_lat - min_lat) / lat_grid_size))
num_lon_cells = int(np.ceil((max_lon - min_lon) / lon_grid_size))
zmax = result['mag'].max()

# Preallocate arrays
data = [None] * len(result['date'].unique())
heatmap_frames = [None] * len(result['date'].unique())
heatmap = np.zeros((num_lat_cells, num_lon_cells), dtype=float)

for frame, (date, frame_data) in enumerate(result.groupby('date')):
    #scaled vs unscaled data
    for _, row in frame_data.iterrows():
        lat_cell, lon_cell = row['cell_no']
        lat_idx = int(lat_cell - min_lat / lat_grid_size)
        lon_idx = int(lon_cell - min_lon / lon_grid_size)
        heatmap[lat_idx, lon_idx] = row['mag']
        
        data[frame] = [
            np.where(heatmap == 0, 0, heatmap),
        ]
        heatmap_frames[frame] = go.Frame(
        data=go.Heatmap(
            z=np.where(heatmap == 0, None, heatmap),
            colorscale='Plotly3',
            zmin=3,
            zmax=zmax,
            x=np.linspace(min_lon, max_lon, num=num_lon_cells),
            y=np.linspace(min_lat, max_lat, num=num_lat_cells),
            colorbar=dict(title='Sum of Values')
        ),
        name=str(date)
    )
    
    heatmap.fill(0)

data = np.array(data)
flattened_data = np.reshape(data, (data.shape[0], data.shape[1], -1))
print(flattened_data.ndim)
squeezed_data = np.squeeze(flattened_data, axis=1)
df = pd.DataFrame(squeezed_data)
data = df.values

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back, :])
    return np.array(X), np.array(Y)

look_back = 24
X, Y = create_dataset(data, look_back)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, data.shape[1])))
model.add(Dense(data.shape[1], activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_split=0.1)

last_values = data[-look_back:]
last_values = last_values.reshape(1, look_back, data.shape[1])
predicted_rows = np.maximum(model.predict(last_values), 0)

for i in range(49):
    last_values = np.append(last_values[:,1:,:], [predicted_rows], axis=1)
    predicted_rows = np.vstack((predicted_rows, model.predict(last_values)))

matrices = predicted_rows.reshape(50, 20, 20)

for i, matrix in enumerate(matrices):
    frame = go.Frame(
        data=[go.Heatmap(
             z=np.where(matrix == 0, None, matrix),
            colorscale='Plotly3',
            zmin=0,
            zmax=1,
        )],
        name=str(i)
    )
    heatmap_frames.append(frame)

fig_cell_grid_usgs = go.Figure(
    data=[go.Heatmap(z=matrices[0], colorscale='Plotly3')],
    layout=go.Layout(
        title='Sum of Magnitude over Time for Each Cell',
        width=700,
        height=700,
        xaxis_title='Longitude',
        yaxis_title='Latitude',
    ),
    frames=heatmap_frames,
)
fig_cell_grid_usgs.update_layout(coloraxis_colorbar=dict(title='Sum of Magnitude'))
fig_cell_grid_usgs.update_layout(
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {"font": {"size": 20}, "prefix": "Date: ", "visible": True, "xanchor": "right"},
            "transition": {"duration": 500, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [{"args": [[f.name], {"frame": {"duration": 500, "redraw": True},
                                       "mode": "immediate"}],
                   "label": f.name,
                   "method": "animate"} for f in heatmap_frames]
    }]
)