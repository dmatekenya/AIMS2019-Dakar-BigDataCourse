import json
import requests
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from textblob import TextBlob
import pandas as pd
import os
import pandas as pd
import matplotlib
import seaborn as sns



def report_basic_data_properties(csv_file=None):
    df = pd.read_csv(csv_file)
    return (len(df.index),list(df.columns.values))




def plot_a_numeric_attribute(csv_file=None, col_to_plot=None, output_plot=None):
    """
    Given a CSV file, read the data using pandas, plot a given column
    and finally save the plot as "png" file.
    DATA FOR TESTING: day2-python-for-data-science/data/townships_with_dist_and_elev.csv
    COLUMN NAME FOR TESTING: 'elev_meters' column
    :param csv_file: File to get data from
    :param col_to_plot: Column name to plot
    :param output_plot: Save output plot to file
    :return:
    """
    # read data into pandas dataframe
    # [YOUR CODE HERE]
    df = pd.read_csv(csv_file)
    # use seaborn to plot distribution of data
    # ax = sns.distplot(ADD YOUR CODE HERE)
    ax = sns.distplot(df[col_to_plot])
    # save plot as png file
    # ax.get_figure().savefig(ADD YOUR CODE HERE)
    ax.get_figure().savefig(output_plot)


def get_weather(api_key=None, city_id=None):
    url = "http://api.openweathermap.org/data/2.5/forecast?id={}&APPID={}".format(city_id, api_key)
    r = requests.get(url)
    json_obj = r.json()
    return json_obj



def compile_weather_forecast(city_name=None,output_csv_file=None):
    """
    Get weather forecasts for Dakar. Please get only TEMPERATURE and HUMIDITY
    Useful Info:
    city_details_file: day2-python-for-data-science/data/city.list.json
    :param your_api_key:
    :param output_csv_file:
    :return:
    """
    # # copy and paste your API key below
    API_KEY = "fb83f086c3c52f113ccb5df5b46d3394"
    jfile = r"C:\Users\AIMSSN\Desktop\AIMSBD4D\day2\data\city.list.json"
    # load city details file
    with open(jfile, encoding="utf8") as f:
        data = json.load(f)
        #inspect the data object above
        print(type(data))
    #use for loop and if statement to find city id
    for country in data:
        city_code = None
        if country["name"] == city_name:
            city_code = country["id"]
            break

    weather_json = get_weather(API_KEY, city_code)
    weather_items = weather_json["list"]
    #data = pd.DataFrame(weather_items)
    #data.head()
    data = []
    for i in weather_items:
        ts=i["dt_txt"]
        temp = i["main"]["temp"]
        hum = i["main"]["humidity"]
        # for rains and clouds, use get() method to
        # retrieve required values
        rains = i.get("rain")
        clouds=i.get("clouds")["all"]
        data_item = {'forecastTime': ts, 'tempF': temp,'humidity': hum, "rain": rains,'cloudsPercent': clouds}
        data.append(data_item)

    data_frame=pd.DataFrame(data)
    data_frame.to_csv(output_csv_file, index=False)



def translate_to_french_for_dunstan(sentence=None):
    words =nltk.word_tokenize(sentence)
    en_fr = {}
    for w in words:
        en_blob = TextBlob(w)
        fr_blob = en_blob.translate(from_lang="en", to='fr')
        en_fr[w] = fr_blob.raw
    return en_fr


def get_name_of_town_with_highest_elevation(csv_file=None,elev_col="elev_metres"):
   df = pd.read_csv(csv_file)
   maxval=max(df[elev_col])
   temp=df.loc[df[elev_col]==maxval]["township"]
   return (temp.values[0])
