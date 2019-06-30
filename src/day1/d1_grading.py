import os
from importlib import reload
import time
import shutil
import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/solution-notebooks/')

try:
    from solution-notebooks import d1_solution
except ImportError:
    pass


DAY = 1
MARKING_SCHEME = {'return_element_of_a_list': 2, 'calculate_average': 4,
                  'concatenate_strings': 2, 'check_if_list_contains_item': 2,
                  'save_list_to_csv_file':4, 'count_number_of_csv_files': 2}


TEST_INPUTS = {'return_element_of_a_list': (['a', 'X', 'z', 10, 30], -1),
               'calculate_average': [i for i in range(10)],
               'concatenate_strings': ('Khama', 'Matekenya'),
               'check_if_list_contains_item': (['Malawi', 'Zambia', 'Mozambique', 'Dunstan'], 'Senegal'),
               'count_number_of_csv_files': '/Users/dmatekenya/Google-Drive/worldbank/freetown/data',
               'save_list_to_csv_file': [["REG_NAME","REG_CODE","DIST_NAME","ADM_STATUS","DIST_CODE"],
               ["Central","2","Ntchisi","Rural","203"]]}


def get_responses_and_record(sol):
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
        elif k == 'count_number_of_csv_files':
            correct = d1_solution.count_number_of_csv_files(input_folder=v)
            output = sol.count_number_of_csv_files(input_folder=v)
            if output == correct:
                total_marks += MARKING_SCHEME[k]
        elif k == 'save_list_to_csv_file':
            out_csv_d = '/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/day1-intro-to-python/data/listToCsvDun.csv'
            out_csv_p = '/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/day1-intro-to-python/data/listToCsvP.csv'

            d1_solution.save_list_to_csv_file(list=v, csvfile_full_path=out_csv_d)
            sol.save_list_to_csv_file(list=v, csvfile_full_path=out_csv_p)

            dfd = pd.read_csv(out_csv_d)
            dfp = pd.read_csv(out_csv_p)

            # check if they are the same
            if dfd.shape[0] == dfp.shape[0]:
                total_marks += MARKING_SCHEME[k]

    return total_marks


def grade_scripts(candidates_solutions_folder=None):

    lst = os.listdir(candidates_solutions_folder)
    res = []

    for l in lst:
        try:
            # get students name
            first = l.split('_')[0]
            last = l.split('_')[1][:-3]

            # copy module to mai folder and rename it
            current_dir = os.path.dirname(os.path.abspath(__file__))
            os.remove(os.path.join(current_dir, 'candidate_solutions.py'))
            new_solutions_script_path = os.path.join(candidates_solutions_folder, l)
            dest_file = os.path.join(current_dir, 'candidate_solutions.py')
            shutil.copy(new_solutions_script_path, dest_file)

            # run script
            import candidate_solutions as sol
            sol = reload(sol)
            score = get_responses_and_record(sol)

            # print results
            print('Participant name: {} {}, score: {}'.format(first, last, score))
            res.append({'firstName': first, 'lastName': last, 'score': score, 'day': DAY,
                        'totalScore': np.sum(list(MARKING_SCHEME.values()))})
        except Exception as e:
            print(e)
            continue

    return res


def main():
    submissions_folder = "/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/exercise-submission/day1"
    d1_scores = "/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/scores/day1.csv"
    grades = grade_scripts(candidates_solutions_folder=submissions_folder)
    df_grades = pd.DataFrame(grades)
    df_grades.to_csv(d1_scores, index=False)

    
if __name__ == '__main__':
    main()
