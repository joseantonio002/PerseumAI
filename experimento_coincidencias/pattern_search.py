import pandas as pd
import pattern_utils
import Pattern as p
import tendencyCalculator as tc
import matplotlib.pyplot as plt
from random import randint
import normalize_utils as nu
from Pattern import Pattern
import network_utils as nw

INCREMENT = 25
TAMANIO_MINIMO_DATAFRAME = 15
MINIMUM_CONFIDENCE = 1.0

def findHistoricPatternsNN(window_width, company_data, models, company_name, patterns_dict, divisions):
    """
    Find patterns through historic data using neural networks
    Args:
        window_width (int): fixed window size for the search
        company_data (dataframe): dataframe containing the company's close prices
        models (dict): dictionary containing types of patterns as keys an the model as value
        company_name (str): name of the company where the search is taking place
        patterns_dict (Dict[]): dictionary containing types of patterns as keys an pattern data as value
    Returns:
        patterns_found (List[Pattern]): list of patterns found
    """
    patterns_found = []
    i = 0
    increment = pattern_utils.getIncrement(window_width)
    company_data = pattern_utils.applySimpleMovingAverage(company_data) 
    while i < len(company_data) - window_width - 1:
        right_window_index = i + window_width
        if right_window_index >= len(company_data):
            break
        sliced_dataframe = company_data.iloc[i:right_window_index]
        normalized_vector = nu.normalizeVector(sliced_dataframe['SMA'].tolist())
        image = nw.pattern_to_image(normalized_vector)
        image_tensor = nw.transform(image)
        predicted, confidence = nw.classify(image_tensor, models)
        if predicted >= 0 and confidence >= pattern_utils.confidenceForPattern(nw.index_pattern[predicted]):
            pattern_type = nw.index_pattern[predicted]
            left_index, right_index = pattern_utils.enhanceDataframeDistancesMean(40, pattern_type, sliced_dataframe['SMA'].tolist(), patterns_dict, divisions)
            dataframe_segment = sliced_dataframe[left_index:right_index]
            longer_dataframe = company_data[i + left_index:] 
            pattern_tendency = tc.findPatternTendency(dataframe_segment, longer_dataframe, pattern_type)
            if pattern_tendency != None:
                new_pattern_Pattern = p.Pattern(pattern_type, pattern_tendency[1], company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], 40, pattern_tendency[2], source="CNN")
                patterns_found.append(new_pattern_Pattern)
            i += right_index
        else:
            i += increment
    return patterns_found

def findHistoricPatternsAutomatic(window_width, company_data, patterns_dictionary, company_name, divisions):
    """Find patterns through historic data  
    Args:  
        window_width (int): fixed window size for the search
        company_data (dataframe): dataframe containing the company's close prices
        atterns_dictionary (Dict[]): dictionary containing types of patterns as keys an pattern data as value
        company_name (str): name of the company where the search is taking place  
    Returns:  
        patterns_found (List[Pattern]): list of patterns found
    """
    patterns_found = []
    i = 0
    separated_execute = False
    company_data = pattern_utils.applySimpleMovingAverage(company_data)
    if ('double_top' in patterns_dictionary.keys() and 'head_and_shoulders' in patterns_dictionary.keys()) or ('double_bottom' in patterns_dictionary.keys() and 'inv_head_and_shoulders' in patterns_dictionary.keys()):
        separated_execute = True
    if separated_execute:
        for key in patterns_dictionary.keys():
            if key == 'rest_normalized':
                continue
            temp_dict = {key: patterns_dictionary[key]}
            i = 0
            while i < len(company_data) - window_width - 1:
                right_window_index = i + window_width
                if right_window_index >= len(company_data):
                    break
                sliced_dataframe = company_data.iloc[i:right_window_index]
                normalized_vector = nu.normalizeVector(sliced_dataframe['SMA'].tolist())
                new_pattern_type, best_distance_found = pattern_utils.findCommonPattern(normalized_vector, temp_dict)
                if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < pattern_utils.acceptanceDistanceForPattern(new_pattern_type):
                    left_index, right_index = pattern_utils.enhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), temp_dict, divisions)
                    dataframe_segment = sliced_dataframe[left_index:right_index] #Esto sin ventana mejorada
                    longer_dataframe = company_data[i + left_index:] #Quitar left_index si no se usa enhanced dataframe
                    pattern_tendency = tc.findPatternTendency(dataframe_segment, longer_dataframe, new_pattern_type)
                    if pattern_tendency != None:
                        new_pattern = p.Pattern(new_pattern_type, pattern_tendency[1], company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], best_distance_found, pattern_tendency[2])
                        patterns_found.append(new_pattern)
                    i += right_index
                else:
                    i += INCREMENT
    else:
        while i < len(company_data) - window_width - 1:
            right_window_index = i + window_width
            if right_window_index >= len(company_data):
                break
            sliced_dataframe = company_data.iloc[i:right_window_index]
            normalized_vector = nu.normalizeVector(sliced_dataframe['SMA'].tolist())
            new_pattern_type, best_distance_found = pattern_utils.findCommonPattern(normalized_vector, patterns_dictionary)
            if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < pattern_utils.acceptanceDistanceForPattern(new_pattern_type):
                left_index, right_index = pattern_utils.enhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), patterns_dictionary, divisions)
                dataframe_segment = sliced_dataframe[left_index:right_index] #Esto sin ventana mejorada
                longer_dataframe = company_data[i + left_index:] #Quitar left_index si no se usa enhanced dataframe
                pattern_tendency = tc.findPatternTendency(dataframe_segment, longer_dataframe, new_pattern_type)
                if pattern_tendency != None:
                    new_pattern = p.Pattern(new_pattern_type, pattern_tendency[1], company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], best_distance_found, pattern_tendency[2])
                    patterns_found.append(new_pattern)
                i += right_index
            else:
                i += INCREMENT
    return patterns_found


def searchPattern(company_dataframe, normalized_vector, patterns_dictionary, company_name):
  new_pattern_type, distance = pattern_utils.findCommonPattern(normalized_vector, patterns_dictionary)
  new_pattern = None
  if distance < pattern_utils.acceptanceDistanceForPattern(new_pattern_type):
    pattern_tendency = tc.findPatternTendency(company_dataframe, company_dataframe, new_pattern_type)
    if pattern_tendency is not None:
      new_pattern = p.Pattern(new_pattern_type, company_dataframe, company_name, str(company_dataframe.iloc[0].name), str(company_dataframe.iloc[-1].name), None, distance, pattern_tendency[2])
  return new_pattern

def findCurrentPatternsAutomatic(company_dataframe, patterns_dictionary, company_name, attempt=3, failure_increment=0.3, 
                                                                                            acceptance_increment=0.1):
  """
    Find patterns in today's stock market
    Args:
        company_dataframe (dataframe): dataframe containing the company's close prices
        patterns_dictionary (Dict[]): dictionary containing types of patterns as keys an pattern data as value
        company_name (str): name of the company where the search is taking place
        attempt (int): number of attempts to find a pattern
        failure_increment (float): percentage of the dataframe to remove when a pattern is not found
        acceptance_increment (float): percentage of the dataframe to remove when a pattern is found
    Returns:
        best_pattern_Pattern (Pattern): the best pattern found
  """
  company_dataframe = pattern_utils.applySimpleMovingAverage(company_dataframe)
  current_best_pattern = None
  while True:
    normalized_vector = nu.normalizeVector(company_dataframe['SMA'].tolist())
    new_pattern = searchPattern(company_dataframe, normalized_vector, patterns_dictionary, company_name)
    if new_pattern is None:
      increment = round(len(company_dataframe.index) * failure_increment)
      company_dataframe = company_dataframe.iloc[increment: len(company_dataframe.index)]
      attempt -= 1
    else:
      if current_best_pattern is None or new_pattern.distance < current_best_pattern.distance:
        current_best_pattern = new_pattern
        increment = round(len(company_dataframe.index) * acceptance_increment)
        company_dataframe = company_dataframe.iloc[increment: len(company_dataframe.index)]
      else:
        break
    if attempt <= 0 or len(company_dataframe.index) < TAMANIO_MINIMO_DATAFRAME:
      break
  return current_best_pattern


def findCurrentPatternsNN(company_dataframe, models, company_name, attempt=3, failure_increment=0.3, 
                                                                                            acceptance_increment=0.1):
  """
    Find patterns in today's stock market using neural networks
    Args:
        company_dataframe (dataframe): dataframe containing the company's close prices
        models (List[CNN]): list of models to use for training
        company_name (str): name of the company where the search is taking place
        attempt (int): number of attempts to find a pattern
        failure_increment (float): percentage of the dataframe to remove when a pattern is not found
        acceptance_increment (float): percentage of the dataframe to remove when a pattern is found
    Returns:
        best_pattern_Pattern (Pattern): the best pattern found
    """
  company_dataframe = pattern_utils.applySimpleMovingAverage(company_dataframe) 
  current_best_pattern = None
  pattern_type = None
  best_pattern_Pattern = None
  while True:
    normalized_vector = nu.normalizeVector(company_dataframe['SMA'].tolist())
    image = nw.pattern_to_image(normalized_vector)
    image_tensor = nw.transform(image)
    predicted, confidence = nw.classify(image_tensor, models)
    new_pattern = None
    if predicted >= 0 and confidence >= 5.0:
        new_pattern = [normalized_vector, confidence, nw.index_pattern[predicted]]
    if new_pattern is None:
      increment = round(len(company_dataframe.index) * failure_increment)
      company_dataframe = company_dataframe.iloc[increment: len(company_dataframe.index)]
      attempt -= 1
    else:
      if current_best_pattern is None or new_pattern[1] > current_best_pattern[1]:
        current_best_pattern = new_pattern
        pattern_type = current_best_pattern[2]
        increment = round(len(company_dataframe.index) * acceptance_increment)
        company_dataframe = company_dataframe.iloc[increment: len(company_dataframe.index)]
      else:
        break
    if attempt <= 0 or len(company_dataframe.index) < TAMANIO_MINIMO_DATAFRAME:
      break
  if current_best_pattern is not None:
    pattern_tendency = tc.findPatternTendency(company_dataframe, company_dataframe, pattern_type)
    if pattern_tendency != None:#tipo patron  dataframe donde se encontró nombcomp  fecha inicial                       fecha final           Tendencia del patrón  distancia  puntos de interés en los que dibujar una línea en el canvas final
       best_pattern_Pattern = [p.Pattern(pattern_type, company_dataframe, company_name, str(company_dataframe.iloc[0].name), str(company_dataframe.iloc[-1].name), None, confidence, pattern_tendency[2])]
  return best_pattern_Pattern
