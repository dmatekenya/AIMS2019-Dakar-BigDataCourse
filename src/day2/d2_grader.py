import os
from importlib import reload
import shutil
import numpy as np
import pandas as pd

DAY = 2
MARKING_SCHEME = {'report_basic_data_properties': 2, 'get_name_of_town_with_highest_elevation': 4,
                  'plot_a_numeric_attribute': 2, 'translate_to_french_for_dunstan': 2,
                  'convert_website_table_to_csv': 5, 'compile_weather_forecast': 5}
TOT_SCORE = np.sum(list(MARKING_SCHEME.values()))
TOT_QNS = len(MARKING_SCHEME)

base_folder = "/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/"
file_basic_prop = os.path.join(base_folder, 'day2-python-for-data-science/data/power-outages.csv')
elev_file = os.path.join(base_folder, 'day2-python-for-data-science/data/townships_with_dist_and_elev.csv')
plot = os.path.join(base_folder, 'day2-python-for-data-science/data/elev_plot.png')

TEST_INPUTS = {'report_basic_data_properties': file_basic_prop,
               'get_name_of_town_with_highest_elevation': elev_file,
               'plot_a_numeric_attribute': (elev_file, 'elev_metres'),
               'translate_to_french_for_dunstan': 'I love red wine',
               'convert_website_table_to_csv': 'https://www.tcsnycmarathon.org/about-the-race/results/overall-men',
               'compile_weather_forecast': "Montgomery County"}


def get_responses_and_record(student_sol=None, sol=None):
    total_marks = 0
    total_graded = 6
    for k, v in TEST_INPUTS.items():
        try:
            if k == 'report_basic_data_properties':
                print()
                print('Grading function: {}'.format(k))
                s = student_sol.report_basic_data_properties(v)
                d = sol.report_basic_data_properties(v)

                # check if its correct
                for i in s:
                    if isinstance(i, int):
                        if i == d[0]:
                            total_marks += 1

                    if isinstance(i, list):
                        if sorted(i) == sorted(d[1]):
                            total_marks += 1

            elif k == 'get_name_of_town_with_highest_elevation':
                print()
                print('Grading function: {}'.format(k))
                correct = sol.get_name_of_town_with_highest_elevation(csv_file=v)
                output = student_sol.get_name_of_town_with_highest_elevation(csv_file=v)
                # check if its correct
                if output == correct:
                    total_marks += MARKING_SCHEME[k]
            elif k == 'plot_a_numeric_attribute':
                print()
                print('Grading function: {}'.format(k))
                student_sol.plot_a_numeric_attribute(csv_file=elev_file, col_to_plot='elev_metres',
                                                     output_file=plot)
                if os.path.exists(plot):
                    total_marks += MARKING_SCHEME[k]

            elif k == 'translate_to_french_for_dunstan':
                print()
                print('Grading function: {}'.format(k))
                correct = sol.translate_to_french_for_dunstan(sentence=v)
                output = student_sol.translate_to_french_for_dunstan(sentence=v)

                if sorted(list(correct.values())) == sorted(list(output.values())):
                    total_marks += MARKING_SCHEME[k]
            elif k == 'convert_website_table_to_csv':
                print()
                print('Grading function: {}'.format(k))
                scraped_csv_file_d = os.path.join(base_folder,
                                                  'day2-python-for-data-science/data/scrapedWebDataDun.csv')
                scraped_csv_file_s = os.path.join(base_folder, 'day2-python-for-data-science/data/scrapedWebDataP.csv')

                sol.convert_website_table_to_csv(output_csv_file=scraped_csv_file_d, url=v)
                student_sol.convert_website_table_to_csv(output_csv_file=scraped_csv_file_s)

                dfd = pd.read_csv(scraped_csv_file_d)
                dfp = pd.read_csv(scraped_csv_file_d)

                if dfd.shape[0] == dfp.shape[0]:
                    total_marks += MARKING_SCHEME[k]
            elif k == 'compile_weather_forecast':
                print()
                print('Grading function: {}'.format(k))
                weather_data_d = os.path.join(base_folder, 'day2-python-for-data-science/data/weatherForecastDun.csv')
                weather_data_p = os.path.join(base_folder, 'day2-python-for-data-science/data/weatherForecastP.csv')

                sol.compile_weather_forecast(output_csv_file=weather_data_d, city_name=v)
                student_sol.compile_weather_forecast(output_csv_file=weather_data_p, city_name=v)

                dfd = pd.read_csv(weather_data_p)
                dfp = pd.read_csv(weather_data_p)

                if dfd.shape[0] == dfp.shape[0]:
                    total_marks += MARKING_SCHEME[k]
        except Exception as e:
            total_graded -= 1
            print('Failed to grade question: {} because of error below'.format(k))
            print(e)
            continue

    return total_marks


def grade_scripts(candidates_solutions_folder=None):
    # load teacher solutions
    current_dir = os.path.dirname(os.path.abspath(__file__))
    instructor_script = os.path.join('/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/solution_notebooks/',
                                     'd2_solution.py')
    shutil.copy(instructor_script, os.path.join(current_dir, 'd2_solution.py'))
    try:
        import d2_solution as sol
        sol = reload(sol)
    except ImportError as e:
        pass

    lst = os.listdir(candidates_solutions_folder)
    res = []
    total_submissions = 0

    for l in lst:
        try:
            if l.endswith('py'):
                # get students name
                print()
                first = l.split('_')[0]
                last = l.split('_')[1][:-3]
                print('========================================')
                print('Grading: {} {}'.format(first, last))
                print('========================================')

                # copy module to mai folder and rename it
                if os.path.exists(os.path.join(current_dir, 'candidate_solutions.py')):
                    os.remove(os.path.join(current_dir, 'candidate_solutions.py'))

                new_solutions_script_path = os.path.join(candidates_solutions_folder, l)
                dest_file = os.path.join(current_dir, 'candidate_solutions.py')
                shutil.copy(new_solutions_script_path, dest_file)

                # run script
                import candidate_solutions as student_sol
                student_sol = reload(student_sol)
                score = get_responses_and_record(student_sol=student_sol, sol=sol)

                # print results
                print('Score: {}/{}, number of questions graded: {}/{}'.format(score[0], TOT_SCORE, score[1], TOT_QNS))
                res.append({'firstName': first, 'lastName': last, 'score': score, 'day': DAY,
                            'totalScore': TOT_SCORE})
        except Exception as e:
            print("GRADING FAILED DUE TO ERROR DESCRIBED BELOW")
            print(e)
            continue
    print()
    print('***********************************************************')
    print('TOTAL PARTICIPANTS GRADED {}'.format(total_submissions))
    print('***********************************************************')
    return res


def main():
    submissions_folder = "/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/exercise-submission/day2"
    d2_scores = "/Users/dmatekenya/Google-Drive/gigs/aims-dakar-2019/scores/day2.csv"
    grades = grade_scripts(candidates_solutions_folder=submissions_folder)
    df_grades = pd.DataFrame(grades)
    df_grades.to_csv(d2_scores, index=False)


if __name__ == '__main__':
    main()
