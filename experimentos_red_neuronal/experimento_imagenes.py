from datetime import date, timedelta
from pattern_utils import loadPatterns
import get_company_data
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import io
import time
import random

from tendencyCalculator import findPatternTendency
from pattern_utils import calculateSimpleMovingAverage
from normalize_utils import normalizeVector
from dtw_applier import comparePatterns, comparePatterns

INCREMENT = 25
SMA_VALUE = 3
MINIMUM_VALUE_TO_APPLY_SMA = 35
TAMANIO_VENTANA = 180
DIVISIONES = [1, 2, 3, 4]
BIG_NUMBER = 99999999

class Pattern:
    """
    Pattern class used to represent all data related to the found patterns
    """
    def __init__(self, pattern_type, dataframe_segment, company_name, starting_date, ending_date, tendency, distance,lenght_pattern, divisions = DIVISIONES, points = None):
        """str: Type of the given pattern"""
        self.pattern_type = pattern_type
        """dataframe: Dataframe where the pattern was found"""
        self.dataframe_segment = dataframe_segment
        """str: Name of the company where the pattern was found"""
        self.company_name = company_name
        """str: Starting date of the pattern found"""
        self.starting_date = starting_date
        """str: Ending date of the pattern found"""
        self.ending_date = ending_date
        """Boolean: tendency of the pattern found (achieved or not)"""
        self.tendency = tendency
        """float: distance between the pattern found and the closest pattern in the dictionary"""
        self.distance = distance
        """Dataframe[]: points of interest to draw a line on the final canvas"""
        self.points = points
        self.length_pattern = lenght_pattern
        self.divisions = divisions

    def __str__(self):
        """Transforms the object to string"""
        return f'[{self.pattern_type}, {self.starting_date}, {self.ending_date}, {self.points}]'


def MIfindCommonPattern(normalized_vector, all_patterns_dictionary):
    """Find the type of pattern for a given vector

        Args:  
            normalized_vector (List[]): previous normalized vector containing prices
            all_patterns_dictionary (Dict{}): dictionary containing pattern types and prices  
        Return:  
            common_pattern_type (str): type of the type for the pattern
            minimum_distance (float): minimum distance found between the best match and the vector
    """
    #print("Finding common pattern for: " + str(normalized_vector))
    #print("All patterns: " + str(all_patterns_dictionary))
    minimun_distance = BIG_NUMBER
    common_pattern_type = 'rest_normalized'
    dict_of_distances = {}
    for pattern_type in all_patterns_dictionary.keys():
        array_of_distances = []
        for single_pattern in all_patterns_dictionary[pattern_type]:
            current_distance = comparePatterns(normalized_vector, single_pattern)
            array_of_distances.append(current_distance)
        if pattern_type != 'rest_normalized':
            array_of_distances = np.array(array_of_distances)
            mean = np.mean(array_of_distances)
            dict_of_distances[pattern_type] = mean

    for pattern_type, distance in dict_of_distances.items():
        if distance < minimun_distance:
            common_pattern_type = pattern_type
            minimun_distance = distance
    return common_pattern_type, minimun_distance

def acceptanceDistanceForPattern(pattern_type, original_mode=True): 
    if original_mode:
      return 40
    distance = None
    if pattern_type == 'double_top' or pattern_type == 'double_bottom':
        distance = 12 # Como estos patrones son más sencillos le pongo una distancia menor de aceptación para que no acepte tanta morralla y los que acepte tengan calidad
    elif pattern_type == 'head_and_shoulders':
        distance = 25 # Como estos patrones son más complejos le pongo una distancia mayor para que sea más sencillo aceptar
    elif pattern_type == 'inv_head_and_shoulders':
        distance = 20
    elif pattern_type == 'ascending_triangle':
        distance = 22
    elif pattern_type == 'descending_triangle':
        distance = 25
    return distance

def MIenhanceDataframeDistancesMean(distance_found, pattern_type, sliced_vector, all_patterns_dictionary, window_divisions):
    minimum_distance = distance_found
    best_segment_i = 0
    best_segment_j = len(sliced_vector) - 1
    for number_of_parts in window_divisions:
        window_size = len(sliced_vector) // number_of_parts
        left_index = 0
        right_index = window_size
        for i in range(number_of_parts):
            split_vector = sliced_vector[left_index:right_index]
            normalized_split_vector = normalizeVector(split_vector)
            array_of_distances = []
            for single_pattern in all_patterns_dictionary[pattern_type]:
                current_distance = comparePatterns(normalized_split_vector, single_pattern)
                array_of_distances.append(current_distance)
            array_of_distances = np.array(array_of_distances)
            mean = np.mean(array_of_distances)
            if mean <= minimum_distance:
                minimum_distance = mean
                best_segment_i = left_index
                best_segment_j = right_index
            left_index = right_index
            right_index += window_size
        if i == window_divisions[len(window_divisions) - 1]: #Si es la ultima parte, cogemos todo hasta donde termine
            right_index = len(sliced_vector) - 1
    return best_segment_i, best_segment_j

def MIfindHistoricPatterns(window_width, company_data, patterns_dictionary, company_name, DIVISIONS, original_mode=True):
    numero_ventanas = 0
    patterns_found = []
    nones = []
    i = 0
    separated_execute = False
    if (company_data.shape[0] > MINIMUM_VALUE_TO_APPLY_SMA):
      company_data = calculateSimpleMovingAverage(company_data, SMA_VALUE)
      company_data = company_data.iloc[SMA_VALUE-1:]
    else:
      company_data = calculateSimpleMovingAverage(company_data, 1) # No le aplica la media movil  
    if ('double_top' in patterns_dictionary.keys() and 'head_and_shoulders' in patterns_dictionary.keys()) or ('double_bottom' in patterns_dictionary.keys() and 'inv_head_and_shoulders' in patterns_dictionary.keys()):
        separated_execute = True
    if separated_execute:
        for key in patterns_dictionary.keys():
            if key == 'rest_normalized' or key == 'general':
                continue
            temp_dict = {key: patterns_dictionary[key]}
            i = 0
            while i < len(company_data) - window_width - 1:
                right_window_index = i + window_width
                if right_window_index >= len(company_data):
                    break
                sliced_dataframe = company_data.iloc[i:right_window_index]
                normalized_vector = normalizeVector(sliced_dataframe['SMA'].tolist())
                new_pattern_type, best_distance_found = MIfindCommonPattern(normalized_vector, temp_dict)
                if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < acceptanceDistanceForPattern(new_pattern_type, original_mode):
                    if new_pattern_type == 'ascending_triangle' or new_pattern_type == 'descending_triangle':
                        DIVISIONS = [1, 2]
                    else:
                        DIVISIONS = DIVISIONES
                    left_index, right_index = MIenhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), temp_dict, DIVISIONS)
                    dataframe_segment = sliced_dataframe[left_index:right_index] #Esto sin ventana mejorada
                    longer_dataframe = company_data[i + left_index:] #Quitar left_index si no se usa enhanced dataframe
                    pattern_tendency = findPatternTendency(dataframe_segment, longer_dataframe, new_pattern_type)
                    if pattern_tendency != None:
                        new_pattern = Pattern(new_pattern_type, dataframe_segment, company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], best_distance_found, len(dataframe_segment.index) , DIVISIONS,pattern_tendency[2])
                        patterns_found.append(new_pattern)
                    else:
                        new_pattern = Pattern('none', dataframe_segment, company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), 'none', best_distance_found, len(dataframe_segment.index), 'none')
                        nones.append(new_pattern)
                    i += right_index
                else:
                    i += INCREMENT
                numero_ventanas += 1

    else:
        while i < len(company_data) - window_width - 1:
            right_window_index = i + window_width
            if right_window_index >= len(company_data):
                break
            sliced_dataframe = company_data.iloc[i:right_window_index]
            normalized_vector = normalizeVector(sliced_dataframe['SMA'].tolist())
            new_pattern_type, best_distance_found = MIfindCommonPattern(normalized_vector, patterns_dictionary)
            if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < acceptanceDistanceForPattern(new_pattern_type, original_mode):
                if new_pattern_type == 'ascending_triangle' or new_pattern_type == 'descending_triangle':
                    DIVISIONS = [1, 2]
                else:
                    DIVISIONS = DIVISIONES
                left_index, right_index = MIenhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), patterns_dictionary, DIVISIONS)
                dataframe_segment = sliced_dataframe[left_index:right_index] #Esto sin ventana mejorada
                longer_dataframe = company_data[i + left_index:] #Quitar left_index si no se usa enhanced dataframe
                pattern_tendency = findPatternTendency(dataframe_segment, longer_dataframe, new_pattern_type)
                if pattern_tendency != None:
                    new_pattern = Pattern(new_pattern_type, dataframe_segment, company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], best_distance_found, len(dataframe_segment.index), DIVISIONS, pattern_tendency[2])
                    patterns_found.append(new_pattern)
                else:
                    new_pattern = Pattern('none', dataframe_segment, company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), 'none', best_distance_found, len(dataframe_segment.index), 'none')
                    nones.append(new_pattern)
                i += right_index
            else:
                i += INCREMENT
            numero_ventanas += 1
    return patterns_found, nones

def guardarImagenPatron(patron):
    output_dir = './collected_data/'
    dpi = 100
    figsize_x = 256 / dpi
    figsize_y = 64 / dpi
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
    ax.plot(patron.dataframe_segment['SMA'].tolist(), color='black', linewidth=1)
    ax.axis('off')
    fig.patch.set_visible(False)
    plt.tight_layout(pad=0)
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    # Load this image into PIL Image object and convert to black and white
    image = Image.open(buf).convert('L')
    # Construct the output filepath
    #starting_date tiene la forma 'YYYY-MM-DD', osea que el nombre seria nombrecompañia_mesinicial_diainicial_mesfinal_diafinal
    numero = random.randint(1, 1000)
    filename = f'{patron.pattern_type} - {TAMANIO_VENTANA} - {patron.divisions} - {patron.company_name} {numero}'
    guardarCSVpatron(patron, numero)
    final_output_dir = os.path.join(output_dir, patron.pattern_type)
    output_filepath = os.path.join(final_output_dir, f"{filename}.png")
    # Save the image to the specified directory
    image.save(output_filepath)

    buf.close()
    plt.close(fig)

def guardarCSVpatron(patron, numero):
    output_dir = './collected_data/'
    filename = f'{patron.pattern_type} - {TAMANIO_VENTANA} - {patron.divisions} - {patron.company_name} {numero}'
    final_output_dir = os.path.join(output_dir, patron.pattern_type)
    output_filepath = os.path.join(final_output_dir, f"{filename}.csv")
    #normalize the data
    normalized_data = normalizeVector(patron.dataframe_segment['SMA'].tolist())
    normalized_dataframe = pd.DataFrame(normalized_data)
    normalized_dataframe.to_csv(output_filepath, sep=',')

def openTxt(text_file):
    """Read the txt file chose by the user"""
    if text_file != None:
        text_file = open(text_file, 'r')
        input_text = text_file.read()
        companies = parseTxt(input_text)
        return companies

def parseTxt(text):
    """Parse a given txt file"""
    text = text.split()
    result_companies = []
    for word in text:
        result_companies.append(word)
    return result_companies    

patrones_a_estudiar = ['descending_triangle']

if __name__ == "__main__":
  print(TAMANIO_VENTANA)
  diccionario_patrones = loadPatterns(15, patrones_a_estudiar)
  fecha_inicio = date(2015, 1, 2)
  fecha_fin = date(2024, 1, 2)
  encontrados_total = []
  none_total = []
  empresas = openTxt('../../output5.txt')
  for EMPRESA in empresas:
    print(f'Procesando empresa {EMPRESA}')
    dataframe = get_company_data.getCompanyDataWithYahoo(EMPRESA, fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d"))
    if dataframe is None or dataframe.empty:
      print(f'No se ha podido descargar los datos de la empresa {EMPRESA} en el rango de fechas {fecha_inicio} - {fecha_fin}')
      continue
    encontrados, none = MIfindHistoricPatterns(TAMANIO_VENTANA, dataframe, diccionario_patrones, EMPRESA, DIVISIONES, False)
    encontrados_total = encontrados_total + encontrados
    none_total = none_total + none
  #for patron in encontrados_total:
    #guardarImagenPatron(patron)
  for patron in none_total:
    guardarImagenPatron(patron)