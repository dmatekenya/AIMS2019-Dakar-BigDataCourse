

# import Python libraries if required
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from textblob import TextBlob
import nltk
import requests
from bs4 import BeautifulSoup

# define a function, please call it: report_basic_data_properties
# the function should take as input a CSV file, call the input
# parameter "csv_file"
# and return the following properties about the data
# 1. Number of rows in the data
# 2. List of column names
# Note that you can return two values in a function
# using a tuple


def report_basic_data_properties(csv_file=None):

    df = pd.read_csv(csv_file)

    return df.shape[0], list(df.columns)


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
    max_val = df[elev_col].max()
    pd_series = df[df[elev_col] == max_val]['township']

    # convert from pandas series to single value
    town_name = pd_series.values[0]

    return town_name


def plot_a_numeric_attribute(csv_file=None, col_to_plot=None, output_file=None):
    """
    Given a CSV file, read the data using pandas, plot a given column
    and finally save the plot as png file.
    DATA FOR TESTING: day2-python-for-data-science/data/townships_with_dist_and_elev.csv
    COLUMN NAME FOR TESTING: 'elev_meters' column
    :param csv_file: File to get data from
    :param col_to_plot: Column name to plot
    :param output_file: Save output plot to file
    :return:
    """

    # read data into pandas dataframe
    df = pd.read_csv(csv_file)

    # use seaborn to plot distribution of data
    ax = sns.distplot(df[col_to_plot])

    # save plot as png file
    ax.get_figure().savefig(output_file)


def translate_to_french_for_dunstan(sentence=None):
    """
    Given a sentence, translate each word in the sentence
    Example: sentence = 'I love you', returns {"I": "je", "love": "amour", "you": "vous"}
    use textblob package (https://textblob.readthedocs.io/en/dev/) and NLTK package
    for this task
    :param sentence: Sentence to translate
    :return:
    """
    # first tokenize the words: split the sentence
    # into words using the NLTK function word_tokenize()
    words = nltk.word_tokenize(sentence)

    # initiate a dictionary object to put in english and French words
    en_fr = {}

    # Now do the translation
    for w in words:
        en_blob = TextBlob(w)

        # use the function translate(from_lang="en", to='fr')
        # on the en_blob object defined above
        fr_blob = en_blob.translate(from_lang="en", to='fr')

        # use function raw on the blob above to get the word as a string
        fr_word = fr_blob.raw

        # put the translated word in the
        # dictionary object en_fr with english
        # as key and corresponding french translation as value
        en_fr[w] = fr_word

    # return the dictionary object
    return en_fr


def get_table_rows_from_webpage(url=None):
    """
    Given a web page, returns rows of a table
    :param url: The website to get the table from
    :return:
    """

    # Open the website using requests, retrieve HTML and create BS object
    url = 'https://www.tcsnycmarathon.org/about-the-race/results/overall-men'
    r = requests.get(url)
    html = r.text
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
        # Extract cell using tble cell HTML tag
        cells = row.find_all('td')

        # Extract text only
        str_cells = str(cells)
        clean_text = BeautifulSoup(str_cells, "lxml").get_text()

        # Remove white spaces-a little convuluted but it works
        clean_text2 = " ".join(clean_text.split())

        # Remove brackts at beginning and end
        clean_text3 = clean_text2[1:-1]

        # Split using comma delimiter
        split_str = clean_text3.split(',')

        # Remove white spaces again
        split_str2 = [i.strip() for i in split_str]
        cleaned_rows.append(split_str2)
    
    return cleaned_rows


def convert_website_table_to_csv(url=None, output_csv_file=None):
    """
    The function scrapes data off the website wih given url
    and saves it into CSV file.
    :param url: Website link to scrape data from
    :param output_csv_file: Full path of CSV file to save data to
    :return: Saves data
    """

    # extract table rows using the function "get_table_rows_from_webpage"
    #  defined above
    tb_rows = get_table_rows_from_webpage(url=url)

    # clean up table rows using "clean_table_rows" function
    clean_tb_rows = clean_table_rows(tb_rows=tb_rows)

    # Column names: note that the first element of the list contains the column names
    # Use list indexing to get the column headers
    colnames = clean_tb_rows[0]

    # Create dataframe
    df_men = pd.DataFrame(data=clean_tb_rows[1:], columns=colnames)

    # save the dataframe to CSV file
    df_men.to_csv(output_csv_file,index=False)


