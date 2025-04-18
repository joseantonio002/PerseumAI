import pattern_search
import get_company_data
import pattern_utils
import sys
import tendencyCalculator as tc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

NUMBER_PATTERNS_TO_LOAD = 15

"""
Main file for programme. This script manages user input and executes the highest-level functions
"""

def trainHistoricDatabase(company, patterns_dictionary, initial_date, final_date, window_size):
    """
    Reads the database and trains the data starting from the given year  

    Args:  
        company (str): Company where we want to train from
        year (int): Year since we want to train  
    Returns:  
        patterns_found (List[Pattern]): A list containing all the patterns that were found
    """
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, initial_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print('empty or None dataframe x2')
        return []
    WINDOW_SIZE, DIVISIONS = pattern_utils.windowSizeAndDivisions(company_dataframe)
    print(WINDOW_SIZE, DIVISIONS)
    patterns_found = pattern_search.findHistoricPatterns(WINDOW_SIZE, company_dataframe, patterns_dictionary, company, DIVISIONS)
    #average_tendency = pattern_utils.calculateTendencyProbability(patterns_found, patterns_dictionary.keys()) SE LLAMA DOS VECES Y SOBRA ESTO CREO
    return patterns_found
#
def findCurrentPatterns(company, patterns_dictionary, window_size):
    """
    Finds if there are patterns in today's stock market  

    Args:  
        company (str): Company where we want to train from
        patterns_to_find List[str]: Type of patterns we want to find today  
    Returns:  
        patterns_found (List[Pattern]): A list containing all the patterns that were found
    """
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print(f'Data for company {company} is empty or None')
        return []
    patterns_found = pattern_search.findCurrentPatterns(company_dataframe, patterns_dictionary, company)
    return patterns_found

if len(sys.argv) > 1:
    patterns_dictionary = pattern_utils.loadPatterns(NUMBER_PATTERNS_TO_LOAD, {'false_positives', 'double_top', 'double_bottom'})
    if sys.argv[1] == '0':
        trainHistoricDatabase(sys.argv[2], sys.argv[3], patterns_dictionary)
    elif sys.argv[1] == '1':
        patterns_found = findCurrentPatterns(sys.argv[2], patterns_dictionary)
        patterns_found[0].dataframe_segment.plot()
        plt.show()
    else:
        exit("Option not valid")