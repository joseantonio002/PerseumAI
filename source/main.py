import pattern_search
import get_company_data
import pattern_utils
import sys
import tendencyCalculator as tc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

WINDOW_SIZE = 130

"""
Main file for programme. This script manages user input and executes the highest-level functions
"""
def trainHistoricDatabaseNetwork(company, patterns_dictionary, initial_date, final_date, models, window_sizes):
    """
    Reads the database and trains the data starting from the given year  

    Args:  
        company (str): Company where we want to train from
        patterns_dictionary (Dict[str, List[Patterns]]): Dictionary containing the patterns we want to find
        initial_date (datetime): Initial date to start training
        final_date (datetime): Final date to stop training
        models (Dict[str, CNN]): dictionary of models to use for training where the key is the pattern type
        window_sizes (List[int]): List of window sizes to use for training 
    Returns:  
        patterns_found (List[Pattern]): A list containing all the patterns that were found
    """
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, initial_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print('empty or None dataframe x2')
        return []
    patterns_found = []
    for window_size in window_sizes:
        divisions = pattern_utils.divisionsForWindowSize(window_size)
        patterns_found = patterns_found + pattern_search.findHistoricPatternsNN(window_size, company_dataframe, models, company, patterns_dictionary, divisions)
    patterns_found = pattern_utils.deleteRepeatedPatterns(patterns_found)
    return patterns_found

def trainHistoricDatabaseAutomatic(company, patterns_dictionary, initial_date, final_date, window_sizes):
    """
    Reads the database and trains the data starting from the given year  

    Args:  
        company (str): Company where we want to train from
        patterns_dictionary (Dict[str, List[Patterns]]): Dictionary containing the patterns we want to find
        initial_date (datetime): Initial date to start training
        final_date (datetime): Final date to stop training
        window_sizes (List[int]): List of window sizes to use for training
    Returns:  
        patterns_found (List[Pattern]): A list containing all the patterns that were found
    """
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, initial_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print('empty or None dataframe x2')
        return []
    patterns_found = []
    for window_size in window_sizes:
        divisions = pattern_utils.divisionsForWindowSize(window_size)
        patterns_found = patterns_found + pattern_search.findHistoricPatternsAutomatic(window_size, company_dataframe, patterns_dictionary, company, divisions)
    patterns_found = pattern_utils.deleteRepeatedPatterns(patterns_found)
    return patterns_found

def findCurrentPatternsAutomatic(company_name, patterns_dictionary, window_sizes):
    """
    Finds if there are patterns in today's stock market  

    Args:  
        company (str): Company where we want to train from
        patterns_dictionary (Dict[str, List[Patterns]]): Dictionary containing the patterns we want to find
        window_sizes (List[int]): List of window sizes to use for training 
    Returns:  
        patterns_found (List[Pattern]): A list containing all the patterns that were found
    """
    patterns_found = None
    for window_size in window_sizes:
        company_dataframe = get_company_data.getCompanyDataWithYahoo(company_name, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
        if company_dataframe is None or company_dataframe.empty:
            print(f'Data for company {company_name} is empty or None')
            continue
        patterns_found = pattern_search.findCurrentPatternsAutomatic(company_dataframe, patterns_dictionary, company_name)
        if patterns_found is not None:
          break
    if patterns_found is None:
      return []
    return [patterns_found]

def findCurrentPatternsNetwork(company_name, models, window_sizes):
    """
    Finds if there are patterns in today's stock market
    Args:
        company_name (str): Company where we want to train from
        models (Dict[str, CNN]): dictionary of models to use for training where the key is the pattern type
        window_sizes (List[int]): List of window sizes to use for training
    Returns:
        patterns_found (List[Pattern]): A list containing all the patterns that were found
    """
    pattern_found = None
    for window_size in window_sizes:
        company_dataframe = get_company_data.getCompanyDataWithYahoo(company_name, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
        if company_dataframe is None or company_dataframe.empty:
            print(f'Data for company {company_name} is empty or None')
            continue
        pattern_found = pattern_search.findCurrentPatternsNN(company_dataframe, models, company_name)
        if pattern_found is not None:
          break
    if pattern_found is None:
        return []
    return pattern_found


if __name__ == "__main__":
    pass