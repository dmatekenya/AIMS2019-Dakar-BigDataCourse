import os
import shutil
import numpy as np
import candidate_solutions as sol

MARKING_SCHEME = {'return_element_of_a_list': 2, 'calculate_average': 4,
                  'concatenate_strings': 2, 'check_if_list_contains_item': 2}

TEST_INPUTS = {'return_element_of_a_list': (['a', 'X', 'z', 10, 30], -1),
               'calculate_average': [i for i in range(10)],
               'concatenate_strings': ('Khama', 'Matekenya'),
               'check_if_list_contains_item': (['Malawi', 'Zambia', 'Mozambique', 'Dunstan'], 'Senegal')}


def get_responses_and_record():
    total_marks = 0
    for k, v in TEST_INPUTS.items():
        # run function
        if k == 'return_element_of_a_list':
            i = v[1]
            lst = v[0]
            output = sol.return_element_of_a_list(lst, i)

            # check if its correct
            if output == lst[i]:
                total_marks += MARKING_SCHEME[k]
        elif k == 'calculate_average':
            output = sol.calculate_average(v)
            # check if its correct
            if output == np.mean(v):
                total_marks += MARKING_SCHEME[k]
        elif k == 'concatenate_strings':
            first = v[0]
            last = v[1]
            output = sol.concatenate_strings(first, last)
            if output == "{}  {}".format(first, last):
                total_marks += MARKING_SCHEME[k]

        elif k == 'check_if_list_contains_item':
            item = v[1]
            lst = v[0]
            output = sol.check_if_list_contains_item(lst, item)
            if item in lst:
                actual = 'YES'
            else:
                actual = 'NO'

            if output == actual:
                total_marks += MARKING_SCHEME[k]

    return total_marks


def mark_all_scripts(candidates_folder=None, results_file=None):

    lst = os.listdir(candidates_folder)
    res = []

    for l in lst:
        try:
            # get students name
            first = l.split('_')[0]
            last = l.split('_')[1]

            # copy module to mai folder and rename it
            full_path = os.path.join(candidates_folder, l)
            shutil.copy(full_path, 'candidate_solutions.py')

            # run script
            res = get_responses_and_record()

            res.append({'firstName': first, 'lastName': last})
        except Exception as e:
            print(e)
            continue

    return res


def main():
    marks = get_responses_and_record()
    print(marks)


if __name__ == '__main__':
    main()
