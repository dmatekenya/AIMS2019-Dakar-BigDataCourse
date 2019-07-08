# use list indexing to get the ith item
def return_element_of_a_list(list_of_things,i):
    return list_of_things[i]
list_elemet =['1','b','d']
   # return the item
return_element_of_a_list(list_elemet,0)

# calcul average
def calculate_average(list_of_numbers):
    num_sum = 0
    for i in list_of_numbers:
        num_sum += i
    n = len(list_of_numbers)
    moyen = num_sum / n
    return moyen
   # Example
list_num = [1, 5, 6, 4]
calculate_average(list_num)

     # concatenate_strings
def concatenate_strings(first_name, last_name):
    name = first_name+' '+last_name
    return name
    # Example
concatenate_strings('Dunstain','Matekenya')

# check list
def check_if_list_contains_item(list_of_things, item):
    if item in list_of_things:
        print('YES')
    else:
        print('NO')
  # Example
check_if_list_contains_item(['a', 'X', 'z'], 'M')

