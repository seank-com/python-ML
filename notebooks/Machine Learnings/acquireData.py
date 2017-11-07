import ipywidgets as widgets
import pandas as pd
import os

choices = {}
other = {}

def test():
    print("hello")
    return "world"

def renderDataSources():
    global choices
    global other

    files = []
    for entry in os.scandir('Data'):
        if entry.name.endswith('.csv') and entry.is_file():
            files.append(entry.name)

    choices = widgets.Dropdown(
        options=files,
        description='Data Set:',
        disabled=False,
    )

    other = widgets.Text(
        value='',
        placeholder='Enter a URL to a csv file',
        description='URL:',
        disabled=False
    )
    return widgets.VBox([choices, other])

def renderData():
    global choices
    global other

    df = {}
    if (other.value != ''):
        df = pd.read_csv(other.value)
#url="https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
    else:
        df = pd.read_csv(os.path.join('Data', choices.value))

    return df
