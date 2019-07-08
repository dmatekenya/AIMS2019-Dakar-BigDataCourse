#!/usr/bin/env python
# coding: utf-8

# In[95]:


# import Python libraries if required
import seaborn as sns
from matplotlib import pyplot as plt
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[96]:


def get_table_rows_from_webpage(url=None):
    """
    The function should go to the webpage given in the parameter
    extract the table values and save to CSV file
    :param url: The website to get the table from
    :return:
    """

    # Open the website using requests, retrieve HTML and create BS object
    # url = "http://www.hubertiming.com/results/2017GPTR10K"
    response = requests.get(url)
    html = response.text
    bs = BeautifulSoup(html, 'lxml')

    # Now get all table rows using the tr tag
    # tb_rows = [YOUR CODE HERE]
    
    tb_rows= bs.find_all('tr')

    # return the table rows
    return tb_rows


# In[151]:



def clean_table_rows(tb_rows=None):
    """
    Since 
    :param tb_rows:
    :return:
    """

    # Declare list to hold all cleaned rows
    cleaned_rows = []

    for row in tb_rows:
         # Extract cell using table cell HTML tag
        cells = row.find_all('td')
    
         # Extract text only
        str_cells = str(cells)
        clean_text = BeautifulSoup(str_cells, "lxml").get_text()
    
            # Remove white spaces-a little convuluted but it works
        clean_text2 = " ".join(clean_text.split())

             # Remove brackts at beginning and end
        clean_text3 = clean_text2[1:-1]

             # Split clean_text3 using comma delimiter
        split_str =  clean_text2.split(',') 


             # Remove white spaces again
        split_str2 = [i.strip() for i in split_str]

             # Add split_str2 to cleaned_rows list
        cleaned_rows.append(split_str2)

    # return cleaned rows
    return cleaned_rows


# In[159]:


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
    tb_rows =  get_table_rows_from_webpage(URL)

    # clean up table rows using "clean_table_rows" function
    clean_tb_rows = clean_table_rows(tb_rows)
        
    # Get Table headers using 'th' HTML tag
"""
  
    response = requests.get(URL)
    html = response.text
    bs = BeautifulSoup(html, 'lxml')
    headers_with_tags = bs.find_all('th')

    # Convert to string
    headers_str = str(headers_with_tags)

    # Extract text only and leave out HTML tags
    headers_without_tags = BeautifulSoup(headers_str, "lxml").get_text()
    headers_without_tags2 = headers_without_tags[1:-1]

    # Split using comma delimeter and remove any trailing spaces
    split_header = headers_without_tags2.split(',')
    split_header2 = [i.strip() for i in split_header] 
       
    # Column names: note that the first element of the list contains the column names
    # Use list indexing to get the column headers
    colnames = split_header2
"""
colnames = clean_tb_rows[0]
            # Create dataframe
df_men = pd.DataFrame(data=clean_tb_rows[1:], columns=colnames)
df_men.head()
print(df_men)
            # save the dataframe to CSV file
            # Save AS CSV into data folder
out_file = '/Users\AIMSSN\Desktop\day2/pi.csv'

        # The index = False option ensures we dont save the default index
df_men.to_csv(out_file, index=False)


# In[158]:


convert_website_table_to_csv()


# In[ ]:




