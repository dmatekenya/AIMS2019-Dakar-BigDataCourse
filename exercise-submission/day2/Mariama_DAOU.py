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

from importlib import reload
import shutil
import numpy as np
import pandas as pd


# define a function, please call it: report_basic_data_properties
# the function should take as input a CSV file, call the input
# parameter "csv_file"
# and return the following properties about the data
# 1. Number of rows in the data
# 2. List of column names
# Note that you can return two values in a function
# using a tuple


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

    lire = pd.read.csv('C:\\Users\\AIMSSN\\Desktop\\BD4D\\day2\\data\\townships_with_dist_and_elev.csv')

    # get the maximum value for elev_metres column
    max = lire.elev_metres.max()

    # inspect the object type which you get above
    # if its a series object use the function
    # "values" on it like so: pd_series.values
    # in order to get a string
    # [YOUR CODE HERE]

    # return the answer
    # [YOUR CODE HERE]


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

    # use seaborn to plot distribution of data
    # ax = sns.distplot(ADD YOUR CODE HERE)

    # save plot as png file
    # ax.get_figure().savefig(ADD YOUR CODE HERE)


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
    # words = [YOUR CODE HERE]

    # initiate a dictionary object to put in english and French words
    en_fr = {}

    # Now do the translation
    # for w in words:
    #     en_blob = TextBlob(w)
    #
    #     # use the function translate(from_lang="en", to='fr')
    #     # on the en_blob object defined above
    #     fr_blob = [YOUR CODE HERE]
    #
    #     # use function raw on the blob above to get the word as a string
    #     [YOUR CODE HERE]
    #
    #     # put the translated word in the
    #     # dictionary object en_fr with english
    #     # as key and corresponding french translation as value
    #     [YOUR CODE HERE]

    # return the dictionary object


def get_table_rows_from_webpage(url=None):
    """
    The function should go to the webpage given in the parameter
    extract the table values and save to CSV file
    :param url: The website to get the table from
    :return:
    """

    # Open the website using requests, retrieve HTML and create BS object
    response = request.get(url)
    html = response
    bs = BeautifulSoup(html, "lxml")

    # Now get all table rows using the tr tag
    tb_rows = BeautifulSoup.find_all('tr')

    # return the table rows
    tb_rows


def clean_table_rows(tb_rows=None):
    """
    Since
    :param tb_rows:
    :return:
    """

    # Declare list to hold all cleaned rows
    cleaned_rows = []

    # for row in tb_rows:
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
cleaned_rows = split_str2

# return cleaned rows
cleaned_rows


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
    tb_rows = "get_table_rows_from_webpage"


# clean up table rows using "clean_table_rows" function
clean_tb_rows = clean_tb_rows(tb_rows)

# Column names: note that the first element of the list contains the column names
# Use list indexing to get the column headers
colnames = clean_tb_rows(0)

# Create dataframe
df_men = pd.DataFrame(data=clean_tb_rows[1:], columns=colnames)

# save the dataframe to CSV file
# [YOUR CODE  HERE]
