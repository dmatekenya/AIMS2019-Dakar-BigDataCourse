"""
GENERAL INSTRUCTIONS
WARNING: For Python beginners:
the instructions here will only make sense after you have gone through and
completed the training materials.
1. Uncomment every line with  [YOUR CODE HERE] and replace it with your code
2. In the functions, I use pass as a place holder, please replace
"""

# import Python libraries if required
import numpy as np
import os


def return_element_of_a_list(list_of_things, i):
    """
    Return the i-th item of the list
    Example:
    Given list ['a', 'X', 'z'] and i: 2, the function returns 'z'
    :param i: Index of the item to return
    :return:
    """
    return list_of_things[i]


def calculate_average(list_of_numbers):
    """
    The function takes in a list of numbers and returns their mean.
    Example
    Given list [1, 2, 3]), the function returns 2
    :param list_of_numbers:A list of numbers
    :return:
    """
    return np.mean(list_of_numbers)


def concatenate_strings(first_name, last_name):
    """
     Given a persons first and last name, return a combined
     full name with space in between them.
     Example
     Given first name: 'Dunstan' and last name 'Matekenya' the function
     returns 'Dunstan Matekenya'. Note that there need to be space in between
    :param first_name: A string variable for first name
    :param last_name: A string variable for first name
    :return:
    """
    # use string concatenation to combine the two strings
    return "{}{}".format(first_name, last_name)


def check_if_list_contains_item(list_of_things, item):
    """
    Given a list, check if item is in the list, return 'YES' if
    thats the case and return NO if not
    Example
    Given list ['a', 'X', 'z'] and item 'M' the function should
    return  'NO'
    :param list_of_things:
    :param item:
    :return:
    """
    if item in list_of_things:
        return 'YES'

    return 'NO'


def count_number_of_csv_files(input_folder=None):
    """
    The function should return number of CSV files in a given folder
    :param input_folder:
    :return:
    """

    # use the os module to list all files in the folder and put them in a list
    file_lst = os.listdir(input_folder)

    # use for loop, list indexing and if conditional statement to get the result
    cnt_csv = 0

    for f in file_lst:
        if f[-3:] == 'csv':
            cnt_csv += 1

    # return the result
    return cnt_csv


def save_list_to_csv(lst=None, csvfile_path=None):
    """
    Given a list (which can be nested), write elements of the list to a CSV file.
    Example
    Given list = ['a', 'X', 'z'], the CSV file will will have one row like this:
    'a', 'X', 'z'
    Given list = [['a', 'X', 'z'], ['1', '2', '3']], CSV file will have two rows
    like below:
    'a', 'X', 'z'
    '1', '2', '3'
    :param lst: List with items to write. Note that it can be a nested list
    :param csvfile_path: full path of CSV file, when testing, dont forget extension
    :return:
    """
    # to avoid some problems later, lets remove the file if it exist
    # use if statement and function 'remove' from os module
    if os.path.exists(csvfile_path):
        os.remove(csvfile_path)

    # open file object for writing
    f = open(csvfile_path, "w+")

    # start writing
    if isinstance(lst[0], list):
        for r in lst:
            item = ','.join(r)
            f.write(item)
            f.write("\n")
    else:
        str = ','.join(lst)
        f.write(str)

    # close file object
    f.close()
