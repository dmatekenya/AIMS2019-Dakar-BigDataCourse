def return_element_of_a_list(list_of_things, i):
    """
    Return the i-th item of the list
    Example:
    Given list ['a', 'X', 'z'] and i: 2, the function returns 'z'
    :param i: Index of the item to return
    :return:
    """
    # use list indexing to get the ith item
    a = list_of_things[i]
    # [YOUR CODE HERE]

    # return the item
    return a
    # [YOUR CODE HERE]
names = ["Boa", "Kevine", "Djavph", "Diouf"]
return_element_of_a_list(names, 2)


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
    for i in list_of_numbers:
        sum += i

    # get the length of the list of numbers using list function len()
    length = len(list_of_numbers)

    # finally calculate the mean
    m = sum / length

    # return the mean
    return m