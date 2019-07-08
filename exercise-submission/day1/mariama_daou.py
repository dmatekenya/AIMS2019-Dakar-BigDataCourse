def return_element_of_a_list(list_of_things, i):
    """
    Return the i-th item of the list
    Example:
    Given list ['a', 'X', 'z'] and i: 2, the function returns 'z'
    :param i: Index of the item to return
    :return:
    """
    # use list indexing to get the ith item
    # [YOUR CODE HERE]
var = list_of_things[i]
return var
    # return the item
    # [YOUR CODE HERE]
    list = [1, 2, 3]
return_element_of_a_list(list,1)



def calculate_average(list_of_numbers):
    """
    The function takes in a list of numbers and returns their mean.
    Example
    Given list [1, 2, 3]), the function returns 2
    :param list_of_numbers:A list of numbers
    :return:
    """

    # calculate  the sum using for loop
    # [YOUR CODE HERE]
    nums = [1, 2, 3]
    num_sum = 0
    for num in nums
        num_sum += num
        print(num_sum)
    # get the length of the list of numbers using list function len()
    # [YOUR CODE HERE]
    len()
    # finally calculate the mean
    # [YOUR CODE HERE]
    mean = num_sum/len()
    # return the mean
    # [YOUR CODE HERE]
    return mean

def concatenate_strings(first_name, last_name):
    first_name = "Mariama"
    last_name = "DAOU"
    mine = first_name + ' ' + last_name
    print(mine)