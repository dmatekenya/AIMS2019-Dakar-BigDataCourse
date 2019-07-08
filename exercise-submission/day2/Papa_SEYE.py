{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "from bs4 import BeautifulSoup\n",
    "import seaborn as sns\n",
    "from matplotlib import pyplot as plt\n",
    "from textblob import TextBlob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_table_rows_from_webpage(url=None):\n",
    "    \"\"\"\n",
    "    The function should go to the webpage given in the parameter\n",
    "    extract the table values and save to CSV file\n",
    "    :param url: The website to get the table from\n",
    "    :return:\n",
    "    \"\"\"\n",
    "\n",
    "    # Open the website using requests, retrieve HTML and create BS object\n",
    "    response = requests.get(url)\n",
    "    html = response.text\n",
    "    bs = BeautifulSoup(html, \"lxml\")\n",
    "\n",
    "    # Now get all table rows using the tr tag\n",
    "    tb_rows = bs.find_all('tr')\n",
    "\n",
    "    # return the table rows\n",
    "    return tb_rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_table_rows(tb_rows=None):\n",
    "    \"\"\"\n",
    "    Since \n",
    "    :param tb_rows:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "\n",
    "    # Declare list to hold all cleaned rows\n",
    "    cleaned_rows = []\n",
    "\n",
    "    for row in tb_rows:\n",
    "    #     # Extract cell using table cell HTML tag\n",
    "        cells = row.find_all('td')\n",
    "    #\n",
    "    #     # Extract text only\n",
    "        str_cells = str(cells)\n",
    "        clean_text = BeautifulSoup(str_cells, \"lxml\").get_text()\n",
    "    #\n",
    "    #     # Remove white spaces-a little convuluted but it works\n",
    "        clean_text2 = \" \".join(clean_text.split())\n",
    "    #\n",
    "    #     # Remove brackts at beginning and end\n",
    "        clean_text3 = clean_text2[1:-1]\n",
    "    #\n",
    "    #     # Split clean_text3 using comma delimiter\n",
    "        split_str = clean_text3.split(',')\n",
    "    #\n",
    "    #\n",
    "    #     # Remove white spaces again\n",
    "        split_str2 = [i.strip() for i in split_str]\n",
    "    #\n",
    "    #     # Add split_str2 to cleaned_rows list\n",
    "        cleaned_rows.append(split_str2)\n",
    "\n",
    "    # return cleaned rows\n",
    "    return cleaned_rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert_website_table_to_csv(output_csv_file=None):\n",
    "    \"\"\"\n",
    "    The function scrapes data off the website wih given url\n",
    "    and saves it into CSV file.\n",
    "    :param output_csv_file:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    # URL to get data from\n",
    "    URL = 'https://www.tcsnycmarathon.org/about-the-race/results/overall-men'\n",
    "\n",
    "    # extract table rows using the function \"get_table_rows_from_webpage\"\n",
    "    #  defined above\n",
    "    tb_rows =  get_table_rows_from_webpage(URL)\n",
    "    # clean up table rows using \"clean_table_rows\" function\n",
    "    clean_tb_rows =  clean_table_rows(tb_rows)\n",
    "\n",
    "    # Column names: note that the first element of the list contains the column names\n",
    "    # Use list indexing to get the column headers\n",
    "    #return clean_tb_rows\n",
    "    \n",
    "    colnames = clean_tb_rows[0]\n",
    "    print(len(colnames))\n",
    "#     # Create dataframe\n",
    "    df_men = pd.DataFrame(data=clean_tb_rows[1:], columns=colnames)\n",
    "\n",
    "#     # save the dataframe to CSV file\n",
    "    df_men.to_csv(output_csv_file)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7\n"
     ]
    }
   ],
   "source": [
    "convert_website_table_to_csv()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7\n"
     ]
    }
   ],
   "source": [
    "convert_website_table_to_csv(\"C:/Users/AIMSSN/Desktop/day2/df_men.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = pd.read_csv(\"C:/Users/AIMSSN/Desktop/day2/df_men.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Place</th>\n",
       "      <th>Bib</th>\n",
       "      <th>Name</th>\n",
       "      <th>Time</th>\n",
       "      <th>State</th>\n",
       "      <th>Country</th>\n",
       "      <th>Citizenship</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>Lelisa Desisa</td>\n",
       "      <td>2:05:59</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Ethiopia</td>\n",
       "      <td>ETH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Shura Kitata</td>\n",
       "      <td>2:06:01</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Ethiopia</td>\n",
       "      <td>ETH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Geoffrey Kamworor</td>\n",
       "      <td>2:06:26</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Kenya</td>\n",
       "      <td>KEN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>Tamirat Tola</td>\n",
       "      <td>2:08:30</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Ethiopia</td>\n",
       "      <td>ETH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>Daniel Wanjiru</td>\n",
       "      <td>2:10:21</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Kenya</td>\n",
       "      <td>KEN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Place  Bib               Name     Time State   Country  \\\n",
       "0           0    1.0  3.0      Lelisa Desisa  2:05:59   NaN  Ethiopia   \n",
       "1           1    2.0  2.0       Shura Kitata  2:06:01   NaN  Ethiopia   \n",
       "2           2    3.0  1.0  Geoffrey Kamworor  2:06:26   NaN     Kenya   \n",
       "3           3    4.0  4.0       Tamirat Tola  2:08:30   NaN  Ethiopia   \n",
       "4           4    5.0  5.0     Daniel Wanjiru  2:10:21   NaN     Kenya   \n",
       "\n",
       "  Citizenship  \n",
       "0         ETH  \n",
       "1         ETH  \n",
       "2         KEN  \n",
       "3         ETH  \n",
       "4         KEN  "
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
