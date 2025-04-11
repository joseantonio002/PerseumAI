import pattern_search
import get_company_data
import pattern_utils
import sys
import tendencyCalculator as tc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pattern_utils import deleteRepeatedPatterns

NUMBER_PATTERNS_TO_LOAD = 15

"""
Main file for programme. This script manages user input and executes the highest-level functions
"""

def trainHistoricDatabaseNetwork(company, patterns_dictionary, initial_date, final_date, window_size, models):
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
    #WINDOW_SIZE, DIVISIONS = pattern_utils.windowSizeAndDivisions(company_dataframe)
    #print(WINDOW_SIZE, DIVISIONS)
    #patterns_found = pattern_search.findHistoricPatterns(WINDOW_SIZE, company_dataframe, patterns_dictionary, company, DIVISIONS)
    window_sizes = [80, 120, 160, 200, 240]
    patterns_found = []
    for window_size in window_sizes:
        patterns_found = patterns_found + pattern_search.MIfindHistoricPatterns_red(window_size, company_dataframe, models, company, patterns_dictionary)
    patterns_found = deleteRepeatedPatterns(patterns_found)
    return patterns_found

def trainHistoricDatabaseAutomatic(company, patterns_dictionary, initial_date, final_date, window_size):
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
    window_sizes = [80, 120, 160, 200, 240]
    patterns_found = []
    for window_size in window_sizes:
        divisions = pattern_utils.divisionsForWindowSize(window_size)
        patterns_found = patterns_found + pattern_search.findHistoricPatternsAutomatic(window_size, company_dataframe, patterns_dictionary, company, divisions)
    patterns_found = deleteRepeatedPatterns(patterns_found)
    return patterns_found

def findCurrentPatterns(company_name, patterns_dictionary, window_size):
    """
    Finds if there are patterns in today's stock market  

    Args:  
        company (str): Company where we want to train from
        patterns_to_find List[str]: Type of patterns we want to find today  
    Returns:  
        patterns_found (List[Pattern]): A list containing all the patterns that were found
 
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print(f'Data for company {company} is empty or None')
        return []
    patterns_found = pattern_search.findCurrentPatterns(company_dataframe, patterns_dictionary, company)
    return patterns_found
    """
    patterns_found = None
    windows_to_explore = [40, 60, 80, 100, 120]
    for window_size in windows_to_explore:
        company_dataframe = get_company_data.getCompanyDataWithYahoo(company_name, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
        if company_dataframe is None or company_dataframe.empty:
            print(f'Data for company {company_name} is empty or None')
            continue
        patterns_found = pattern_search.findCurrentPatterns(company_dataframe, patterns_dictionary, company_name)
        if patterns_found is not None:
          break
    if patterns_found is None:
      return []
    return patterns_found

def findCurrentPatternsNetwork(company_name, models):
  patterns_found = None
  windows_to_explore = [40, 60, 80, 100, 120]
  for window_size in windows_to_explore:
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company_name, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print(f'Data for company {company_name} is empty or None')
        continue
    pattern_found = pattern_search.findCurrentPatternsNetwork(company_dataframe, models, company_name)
    if pattern_found is not None:
      break
  if pattern_found is None:
    return []
  return pattern_found


if len(sys.argv) > 1:
    patterns_dictionary = pattern_utils.loadPatterns(NUMBER_PATTERNS_TO_LOAD, {'false_positives', 'double_top', 'double_bottom'})
    if sys.argv[1] == '0':
        trainHistoricDatabaseNetwork(sys.argv[2], sys.argv[3], patterns_dictionary)
    elif sys.argv[1] == '1':
        patterns_found = findCurrentPatterns(sys.argv[2], patterns_dictionary)
        patterns_found[0].dataframe_segment.plot()
        plt.show()
    else:
        exit("Option not valid")