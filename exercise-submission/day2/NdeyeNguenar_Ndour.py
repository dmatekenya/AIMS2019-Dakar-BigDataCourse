"""
GENERAL INSTRUCTIONS
WARNING: For Python beginners:
the instructions here will only make sense after you have gone through and
completed the training materials.

1. WHICH PART TO CHANGE?: Uncomment every line with  [YOUR CODE HERE] and replace it with your code.
Please don't change anything else other than these lines.

2. USE OF JUPYTER NOTEBOOK: For those who would like to use Jupyter Notebook. You can copy and paste
each function in the notebook environment, test your code their. However,
remember to paste back your code in a .py file and ensure that its running
okay.

3. IDENTATION: Please make sure that you check your identation

4. Returning things frm function: All the functions below have to return a value.
Please dont forget to use the return statement to return a value.

5. HINTS: please read my comments for hints and instructions where applicable

6. DEFINING YOUR OWN FUNCTIONS: where I ask you to define your own function
please make sure that you name the function exactly as I said.
"""

# import Python libraries if required
import seaborn as sns
from matplotlib import pyplot as plt
from textblob import TextBlob

# define a function, please call it: report_basic_data_properties
# the function should take as input a CSV file, call the input
# parameter "csv_file"
# and return the following properties about the data
# 1. Number of rows in the data
# 2. List of column names
# Note that you can return two values in a function
# using a tuple

import pandas as pd
import seaborn as sns
from textblob import TextBlob
from nltk import word_tokenize
import requests
from bs4 import BeautifulSoup


def report_basic_data_properties(csv_file=None):
    df = pd.read_csv(csv_file)
    cols = list(df.columns)
    rows = len(df.index)
    values = (rows, cols)
    return values


def get_name_of_town_with_highest_elevation(csv_file=None, elev_col="elev_metres"):
    """
    Given the following data file: day2-python-for-data-science/data/townships_with_dist_and_elev.csv
    return the town with highest elevation.
    Note that column name with elevation values is already provided as a default parameter.
    :param csv_file: CSV file with elevation data
    :param elev_col: Column with elevation values
    :return: town name with highest elevation value
    """
    # read data into pandas dataframe
    df = pd.read_csv(csv_file)

    # get the maximum value for elev_metres column
    max_value = df[df[elev_col] == max(df[elev_col])]

    v = max_value['township']

    # inspect the object type which you get above
    # if its a series object use the function
    # "values" on it like so: pd_series.values
    # in order to get a string
    print(type(v))

    # return the answer
    return str(v.values)


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
    df = pd.read_csv(csv_file)

    # use seaborn to plot distribution of data
    ax = sns.distplot(df[col_to_plot])

    # save plot as png file
    ax.get_figure().savefig(output_plot)


def translate_to_french_for_dunstan(sentence=None):
    """
    Given a sentence, translate each word in the sentence
    Example: sentence = 'I love you', returns {"I": "je", "love": "amour", "you": "vous"}
    use textblob package (https://textblob.readthedocs.io/en/dev/) and NLTK package
    for this task
    :param sentence: Sentence to translate
    :return: a dictionary where key is english word and value is translated french word
    """
    # first tokenize the words: split the sentence
    # into words using the NLTK function word_tokenize()
    words = word_tokenize(sentence)
    print(words)

    # initiate a dictionary object to put in english and French words
    en_fr = {}

    # Now do the translation
    for w in words:
        en_blob = TextBlob(w)
        print(en_blob)

        #     # use the function translate(from_lang="en", to='fr')
        #     # on the en_blob object defined above
        fr_blob = en_blob.translate(to='fr')
        print(fr_blob)
        #
        #     # use function raw on the blob above to get the word as a string

        str_fr = fr_blob.raw
        str_en = en_blob.raw

        #
        #     # put the translated word in the
        #     # dictionary object en_fr with english
        #     # as key and corresponding french translation as value
        en_fr[str_en] = str_fr

    return en_fr


def get_table_rows_from_webpage(url=None):
    """
    The function should go to the webpage given in the parameter
    extract the table values and save to CSV file
    :param url: The website to get the table from
    :return:
    """

    # Open the website using requests, retrieve HTML and create BS object
    response = requests.get(url)
    html = response.text
    bs = BeautifulSoup(html, 'lxml')

    # Now get all table rows using the tr tag
    tb_rows = bs.find_all('tr')

    # return the table rows
    return tb_rows


def clean_table_rows(tb_rows=None):
    """
    Since 
    :param tb_rows:
    :return:
    """

    # Declare list to hold all cleaned rows
    cleaned_rows = []

    for row in tb_rows:
        #     # Extract cell using table cell HTML tag
        cells = row.find_all('td')
        #
        #     # Extract text only
        str_cells = str(cells)
        clean_text = BeautifulSoup(str_cells, "lxml").get_text()
        #
        #     # Remove white spaces-a little convuluted but it works
        clean_text2 = " ".join(clean_text.split())
        #
        #     # Remove brackts at beginning and end
        clean_text3 = clean_text2[1:-1]
        #
        #     # Split clean_text3 using comma delimiter
        split_str = clean_text3.split(',')
        #
        #
        #     # Remove white spaces again
        split_str2 = [i.strip() for i in split_str]
        #
        #     # Add split_str2 to cleaned_rows list
        cleaned_rows.append(split_str2)

    # return cleaned rows
    return cleaned_rows


def convert_website_table_to_csv(output_csv_file=None):
    """
    The function scrapes data off the website wih given url
    and saves it into CSV file.
    :param output_csv_file:
    :return:
    """

    # URL to get data from
    URL = 'https://www.tcsnycmarathon.org/about-the-race/results/overall-men'

    # extract table rows using the function "get_table_rows_from_webpage"
    #  defined above
    tb_rows = get_table_rows_from_webpage(URL)

    # clean up table rows using "clean_table_rows" function
    clean_tb_rows = clean_table_rows(tb_rows)

    # Column names: note that the first element of the list contains the column names
    # Use list indexing to get the column headers

    colnames = clean_tb_rows[0]
    print(colnames)

    # Create dataframe
    df_men = pd.DataFrame(data=clean_tb_rows[1:], columns=colnames)

    # save the dataframe to CSV file
    df_men.to_csv(output_csv_file, index=False)


def get_weather(api_key=None, city_id=None):
    """
    Returns weather
    :param api_key:
    :param city_name:
    :return:
    """
    # add your API key

    url = "http://api.openweathermap.org/data/2.5/forecast?id={}&APPID={}".format(city_id,api_key)

    # use requests to retrieve data from the API
    r = requests.get(url)

    # retrieve JSON from the response object
    dict_obj_from_json = r.json()

    # return the JSON object
    return dict_obj_from_json


def compile_weather_forecast(city_name=None, output_csv_file=None):
    """
    Get weather forecasts for Dakar. Please get only TEMPERATURE and HUMIDITY
    Useful Info:
    city_details_file: day2-python-for-data-science/data/city.list.json
    :param your_api_key:
    :param output_csv_file:
    :return:
    """
    # # copy and paste your API key below
    # API_KEY = [YOUR CODE HERE]
    #
    # # JSON file with city details
    # jfile = [YOUR CODE HERE]
    #
    # # load city details file
    # with open(jfile) as f:
    #     data = json.load(f)
    #
    # # inspect the data object above
    # # use for loop and if statement to find city id
    # city_code = None
    # [YOUR CODE HERE]
    #
    # # now get the weather forecast using the
    # # "get_weather" function defined above
    # weather_json = [YOUR CODE HERE]
    #
    # # using method for accessing a dictionary
    # # put weather items in a list
    # weather_items = [YOUR CODE HERE]
    #
    # # save into a dataframe
    # data = []  # will hold our data
    #
    # for i in weather_items:
    #     # get forecast time
    #     ts = [YOUR CODE HERE]
    #
    #     # get temperature, rain and humidity
    #     temp = [YOUR CODE HERE]
    #     hum = [YOUR CODE HERE]
    #
    #     # for rains and clouds, use get() method to
    #     # retrieve required values
    #     rains = [YOUR CODE HERE]
    #
    #     clouds = [YOUR CODE HERE]
    #
    #     data_item = {'forecastTime': [YOUR CODE HERE], 'tempF': [YOUR CODE HERE],
    #                  'humidity': [YOUR CODE HERE], "rain": [YOUR CODE HERE],
    #                  'cloudsPercent': [YOUR CODE HERE]}
    #
    #     # append to list of create earlier on
    #     [YOUR CODE HERE]
    #

    # # create dataframe
    # [YOUR CODE HERE]
    # 
    # # save dataframe with option index set to False
    # [YOUR CODE HERE]
