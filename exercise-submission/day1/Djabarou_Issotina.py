def calculate_average(list_of_numbers):
    sum_of_numbers=0
    length_of_list=len(list_of_numbers)
    for r in range(length_of_list):
        sum_of_numbers=sum_of_numbers+list_of_numbers[r]
    mean_of_list=sum_of_numbers/length_of_list
    print('The average is {}.'.format(mean_of_list))

def concatenate_strings(first, last):
      return first+" "+last

def check_if_list_contains_item(list_of_things, item):
    second_list=[]
    for i in range(len(list_of_things)):
        if list_of_things[i]==item:
            second_list.append("Yes")
        else:
            second_list.append("No")
    return second_list
