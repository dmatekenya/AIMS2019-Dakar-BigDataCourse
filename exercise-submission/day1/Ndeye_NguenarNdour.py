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
"""


# import Python libraries if required

def return_element_of_a_list(list_of_things, i):
    """
    Return the i-th item of the list
    Example:
    Given list ['a', 'X', 'z'] and i: 2, the function returns 'z'
    :param i: Index of the item to return
    :return:
    """
   # use list indexing to get the ith item
    elt = list_of_things[i]
    
    # return the item
    return elt


def calculate_average(list_of_numbers):
    """
    The function takes in a list of numbers and returns their mean.
    Example
    Given list [1, 2, 3]), the function returns 2
    :param list_of_numbers:A list of numbers
    :return:
    """

    # calculate  the sum using for loop
    sum = 0
    for elt in list_of_numbers:
        sum = sum + elt

    # get the length of the list of numbers using list function len()
    size = len(list_of_numbers)

    # finally calculate the mean
    average = sum / size
    
    # return the mean
    return average


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
    combined_name = first_name + " "+ last_name
    
    return combined_name


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

     # use if and other conditional statements to check for membership
    result = "NO"
    if item in list_of_things:
        result = "YES"

    # return your result
    return result


def count_number_of_csv_files(input_folder=None):
    """
    The function should return number of CSV files in a given folder
    :param input_folder:
    :return:
    """

    import os
    # use the os module to list all files in the folder and put them in a list
    files = os.listdir(input_folder)

    # use for loop, list indexing and if conditional statement to get the result
    csv_number = 0
    for file in files:
        print(file[-3:])
        if file[-3:] == 'csv':
            csv_number = csv_number +1
            
  
    # return the result
    return csv_number


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
    import os
    # to avoid some problems later, lets remove the file if it exist
    # use if statement and function 'remove()' from os module to achiev this
    #if csvfile_path.exists():
       #os.remove(csvfile_path)
 
    # open file object for writing with option "w+"
    fopen_for_writing = open(csvfile_path,'w+')

    # how you write depends on whether the list is nested or not
    # to check is list is nested, use function "isinstance()" to check
    # if elements of the list are also lists
    if(isinstance(lst, list) == True):
        for elt in lst:
            if(isinstance(elt,list) == True):
                fopen_for_writing.write(elt)
            else:
                fopen_for_writing.write(lst)
                break
        

    # close file object
    fopen_for_writing.close()
