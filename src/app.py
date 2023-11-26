'''
 # @ Create Time: 2023-11-26 14:46:34.011634
'''
# Importing the necessary files
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
from dash import Dash, dcc, html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__, title="MyVisual")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Data loading and preprocessing
stock_data = pd.read_csv("data.csv")
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'].str.replace(',', ''), errors='coerce')

# Convert 'Date' column to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

current_time_str = stock_data['Date'].max()
print(current_time_str)
current_time = current_time_str.to_pydatetime()
print(current_time)

# Create a dropdown filter for company selection
companies = stock_data['Name'].unique().tolist()

# Calculate VWAP for each company
stock_data['VWAP'] = (stock_data['Closing_Price'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()

# Calculate Simple Moving Average (SMA) for Closing_Price (taking a window of 3 days as an example)
stock_data['SMA'] = stock_data['Closing_Price'].rolling(window=3).mean()

# Calculate Bollinger Bands
window = 20  # You can adjust the window size as needed
stock_data['Middle_Band'] = stock_data['Closing_Price'].rolling(
    window=window).mean()
stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()
stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()

# Calculate RSI for Closing_Price (taking a window of 14 days as an example)
delta = stock_data['Closing_Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Pivot the stock_dataset to have Date as the index and companies as columns
pivot_stock_data = stock_data.pivot(index='Date',
                                    columns='Name',
                                    values='Closing_Price')

# Calculate correlation matrix
correlation_matrix = pivot_stock_data.corr()

# Calculate quartiles for correlation coefficients
coefficients = correlation_matrix.values.flatten()
lower_quantile = np.percentile(coefficients, 25)
upper_quantile = np.percentile(coefficients, 75)

# Define contours for lower and upper quartiles
contour_lower = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=-1, end=lower_quantile,
                  size=0.05),  # Adjust size as needed
    colorscale='Blues',  # Choose a colorscale for lower quartile contours
    showscale=False,
    name=f'Lower Quartile Contours (-1 - {lower_quantile:.2f})')

contour_upper = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=upper_quantile, end=1,
                  size=0.05),  # Adjust size as needed
    colorscale='Reds',  # Choose a colorscale for upper quartile contours
    showscale=False,
    name=f'Upper Quartile Contours ({upper_quantile:.2f} - 1)')
# Create an empty figure
fig = go.Figure()
fig_rs = go.Figure()
fig_sm = go.Figure()
fig_cs = go.Figure()
fig_bc = go.Figure()
fig_ts = go.Figure()
fig2 = go.Figure()

# Define your layout for each visualization

# Visualization 2 layout
ts_layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown_ts',
        options=[{
            'label': company,
            'value': company
        } for company in companies],
        value=companies[0]  # Set default value to the first company
    ),
    dcc.Graph(figure=fig_ts, id='line-chart')
])

# Combine all visualizations vertically
app.layout = html.Div([
    html.H1("Interactive charts"), ts_layout
])

# Define callback functions for interactivity

@app.callback(
  Output('line-chart', 'figure'),
  [Input('company-dropdown_ts', 'value')]
)
def update_ts_chart(selected_company):
  selected_company_data = stock_data[stock_data['Name'] == selected_company]

  fig_ts = go.Figure()

  for company in companies:
      if company == selected_company:
          fig_ts.add_trace(go.Scatter(x=selected_company_data['Date'],
                                   y=selected_company_data['Closing_Price'],
                                   mode='lines',
                                   name=company,
                                   visible=True))
      else:
          fig_ts.add_trace(go.Scatter(x=[],
                                   y=[],
                                   mode='lines',
                                   name=company,
                                   visible=False))

  fig_ts.update_layout(title='Interactive Time-Series Line Chart For Stock Closing Prices',
                    xaxis_title='Date',
                    yaxis_title='Closing Stock Price')

  return fig_ts

# Run the app
if __name__ == '__main__':
  app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)))