#!/usr/bin/env python
# -*- coding:utf-8 -*-


# import  os,warnings and pandas libraries
import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import numpy as np

warnings.filterwarnings('ignore')
sarima_irl = SARIMAXResults.load(r'sarima_irl.pkl')
sarima_usa = SARIMAXResults.load(r'sarima_usa.pkl')


def process_irl():
    df_irl1 = pd.read_csv(r'Average price 1970 -2001.csv')
    df_irl2 = pd.read_csv(r'Average price 2001-2011.csv')
    df_irl3 = pd.read_csv(r'Average price 2011-2022.csv')
    df_irl3 = df_irl3[df_irl3['TLIST(M1)'] != 201112]

    dfs = [df_irl1, df_irl2, df_irl3]

    def steaks(lists):
        column_1 = lists.drop_duplicates(subset=['Consumer Item'])  # drop duplicated values
        column_1 = column_1[column_1['Consumer Item'].str.contains('steak')]  # keep only observations that contain string 'steak'
        column_1 = column_1['Consumer Item']  # keep only 'Consumer Item' column

    # rename column in df_irl3 to match column names in dfs 1&2
    df_irl3['Consumer Item'] = df_irl3['Consumer Item'].replace(['Sirloin steak per kg'], 'Sirloin steak per kg.')
    # create blank list
    lst = []
    # use for loop  to itterate through all 3 IRL datasets
    for lists in dfs:
        lists = lists.rename(columns={'Consumer Item': 'SteakType', 'TLIST(M1)': 'Daystamp'})  # rename columns for easier manipulation
        lists = lists.loc[(lists.SteakType.str.contains('Sirloin'))]  # keep only observations that contain string 'Sirloin'
        lists = lists.pivot_table('VALUE', ['Daystamp'], 'SteakType')  # use pivot table function to create variables from observations
        lists = lists.rename_axis('ID', axis=1)  # add variable name to ID column
        lists = lists.reset_index()  # reset index- daystamp don't appear if don't reset index
        lst.append(lists)  # apend observations into list
    df_irl = pd.concat(lst, axis=0, ignore_index=True)  # create dataframe from the created list
    df_irl['Daystamp'] = pd.to_datetime(df_irl['Daystamp'].apply(str) + '01', format='%Y%m%d', errors='ignore')
    df_irl['Year'] = df_irl['Daystamp'].dt.year  # Creating 'Year' column that will be used in visualisations
    df_irl['Month'] = df_irl['Daystamp'].dt.month  # Creating 'Month' column that will be used in visualisations
    df_irl.set_index('Daystamp', inplace=True)
    df_irl.resample('M').asfreq()
    df_irl.drop(df_irl.tail(2).index, inplace=True)  # drop last two rows- to have end of year data

    df_usa = pd.read_csv(r'US Steak prices.csv', skiprows=9)
    df_usa['Period'] = df_usa['Period'].str[1:]  # remove 'M' character from 'Period' column
    df_usa['Year'] = df_usa['Year'].astype(str) + df_usa["Period"]  # merge columns 'Year' and 'Period'
    df_usa = df_usa.drop(['Series ID', 'Unnamed: 4', 'Unnamed: 5', 'Period'], axis=1)  # drop unused columns
    df_usa = df_usa.rename(columns={'Year': 'Daystamp'})  # remane 'Year' column to mach IRL dataset
    df_usa = df_usa.rename_axis('ID', axis=1)  # Add column name to the index column
    df_usa['Daystamp'] = pd.to_datetime(df_usa['Daystamp'].apply(str) + '01', format='%Y%m%d', errors='ignore')
    df_usa['Year'] = df_usa['Daystamp'].dt.year  # Creating 'Year' column that will be used in visualisations
    df_usa['Month'] = df_usa['Daystamp'].dt.month  # Creating 'Month' column that will be used in visualisations
    df_usa.set_index('Daystamp', inplace=True)
    df_usa = df_usa.rename(columns={'Value': 'Sirloin steak per kg.'})  # rename 'Value' column to match IRL dataset
    df_usa['Sirloin steak per kg.'] = 2.2 * df_usa['Sirloin steak per kg.']  # convert price from pounds into kilograms
    df_usa = df_usa[(df_usa.index > '1996-12-1') & (df_usa.index <= '2021-12-1')]
    grp = df_usa.groupby('Year')['Sirloin steak per kg.']
    df_usa['Sirloin steak per kg.'] = (
        df_usa['Sirloin steak per kg.'].where(grp.transform('quantile', q=0.95) > df_usa['Sirloin steak per kg.'],
                                              grp.transform('median')))
    return df_irl, df_usa


d1 = {a: {'label': str(a)} for a in range(1997, 2022)}
# d2 = {2022: {'label': "All"}}
d1[int("2022")] = {'label': "All"}

df_irl, df_usa = process_irl()
graph_config = dict(
    {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom", "zoomin", "zoomout"],
    }
)

app = Dash(__name__)
app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='graph1', config=graph_config),
            style={'display': 'inline-block', 'width': '49%'}
        ),
        dbc.Col(
            dcc.Graph(id='graph2', config=graph_config),
            style={'display': 'inline-block', 'width': '49%'}
        ),

    ]),
    dcc.Slider(1997, 2022, 1,
               value=1997,
               id='slider',
               marks=d1
               ),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='graph3', config=graph_config),
            style={'display': 'inline-block', 'width': '49%'}
        ),
        dbc.Col(
            dcc.Graph(id='graph4', config=graph_config),
            style={'display': 'inline-block', 'width': '49%'}
        ),

    ])
])


@app.callback(Output('graph1', 'figure'),
              Output('graph2', 'figure'),
              Output('graph3', 'figure'),
              Output('graph4', 'figure'),
              Input('slider', 'value')
              )
def plot_graph(slider_value):
    df = df_irl[df_irl.index.year == slider_value]
    if slider_value == 2022:
        df = df_irl
    start = list(np.where(df_usa.index == df.index[10]))[0][0]
    predictions_irl = sarima_irl.predict(start=start, end=start + len(df), dynamic=False, type='level')
    df_predict_irl = pd.DataFrame({'Date': predictions_irl.index, 'predicted': predictions_irl.values})
    fig = px.line(df, x=df.index, y="Sirloin steak per kg.", line_shape='spline')
    fig.update_layout(title={'text': 'Sirloin steak price in Ireland in ' + str(slider_value), 'x': 0.5, 'xanchor': 'center'})
    fig2 = px.line(df_predict_irl, x='Date', y="predicted", line_shape='spline')
    fig2.update_layout(
        title={'text': 'Predicted Sirloin steak price in Ireland by Sarima model in ' + str(slider_value), 'x': 0.5, 'xanchor': 'center'})

    df2 = df_usa[df_usa.index.year == slider_value]
    if slider_value == 2022:
        df2 = df_usa
    start = list(np.where(df_usa.index == df2.index[10]))[0][0]
    predictions_usa = sarima_usa.predict(start=start, end=start + len(df2))
    df_predict_usa = pd.DataFrame({'Date': predictions_usa.index, 'predicted': predictions_usa.values})

    fig3 = px.line(df2, x=df2.index, y="Sirloin steak per kg.", line_shape='spline')
    fig3.update_layout(title={'text': 'Sirloin steak price in USA in ' + str(slider_value), 'x': 0.5, 'xanchor': 'center'})
    fig4 = px.line(df_predict_usa, x='Date', y="predicted", line_shape='spline')
    fig4.update_layout(title={'text': 'Predicted Sirloin steak price in USA by Sarima model in ' + str(slider_value), 'x': 0.5, 'xanchor': 'center'})

    return fig, fig2, fig3, fig4


if __name__ == '__main__':
    app.run_server(port=8080, debug=True)
