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

from tendencyCalculator import findPatternTendency
from pattern_utils import calculateSimpleMovingAverage
from normalize_utils import normalizeVector
from pattern_utils import findCommonPattern
from dtw_applier import comparePatterns

INCREMENT = 25
SMA_VALUE = 3
MINIMUM_VALUE_TO_APPLY_SMA = 85 # Aproximadamente 3 meses

def calcular_tiempo(rango):
    fecha_inicial, fecha_final = rango
    
    # Calcula la diferencia entre las fechas
    diferencia = fecha_final - fecha_inicial
    
    # Si la diferencia es de menos de un año
    if diferencia.days < 365:
        # Calcula el número de meses y días
        meses = diferencia.days // 30
        dias = diferencia.days % 30
        
        if meses == 0:
            return f"{dias} días"
        elif dias == 0:
            return f"{meses} meses"
        else:
            return f"{meses} meses y {dias} días"
    else:
        # Calcula el número de años, meses y días
        años = diferencia.days // 365
        meses = (diferencia.days % 365) // 30
        dias = (diferencia.days % 365) % 30
        
        if meses == 0 and dias == 0:
            return f"{años} años"
        elif meses == 0:
            return f"{años} años y {dias} días"
        elif dias == 0:
            return f"{años} años y {meses} meses"
        else:
            return f"{años} años, {meses} meses y {dias} días"

class Pattern:
    """
    Pattern class used to represent all data related to the found patterns
    """
    def __init__(self, pattern_type, dataframe_segment, company_name, starting_date, ending_date, tendency, distance,lenght_pattern ,points = None):
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

    def __str__(self):
        """Transforms the object to string"""
        return f'[{self.pattern_type}, {self.starting_date}, {self.ending_date}, {self.points}]'
    
def MIenhanceDataframeDistancesMean(distance_found, pattern_type, sliced_vector, all_patterns_dictionary, window_divisions, indice_tabla):
    """Given a pattern, find a better match, if possible, inside the vector  

        Args:  
            distance_found (float): minimum distance found between the best match and the vector at the moment
            pattern_type (str): type of the pattern found
            sliced_vector (List[]): vector containing the data where the search will take plave
            all_patterns_dictionary (Dict{}): dictionary containing pattern types and prices
            windows_divisions (List[]): list contaning the number that the window is wanted to be fragmented equally  
        Return:  
            best_segment_i (int): index where the best segment starts
            best_segment_j (int): index where the best segment ends
    """
    minimum_distance = distance_found
    best_segment_i = 0
    best_segment_j = len(sliced_vector) - 1
    distances_list = []
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

            distances_list.append(mean)

            if mean <= minimum_distance:
                minimum_distance = mean
                best_segment_i = left_index
                best_segment_j = right_index
            left_index = right_index
            right_index += window_size
        if i == window_divisions[len(window_divisions) - 1]: #Si es la ultima parte, cogemos todo hasta donde termine
            right_index = len(sliced_vector) - 1
    #with open('distances.csv', 'a') as f:
        #f.write(f'{indice_tabla},{pattern_type},{distances_list},{best_segment_j - best_segment_i}\n')
    return best_segment_i, best_segment_j

def MIfindHistoricPatterns(window_width, company_data, patterns_dictionary, company_name, DIVISIONS, indice_tabla = -1 ):
    """Find patterns through historic data  
    Args:  
        window_width (int): fixed window size for the search
        company_data (datarame): dataframe containing the company's close prices
        atterns_dictionary (Dict[]): dictionary containing types of patterns as keys an pattern data as value
        company_name (str): name of the company where the search is taking place  
    Returns:  
        patterns_found (List[Pattern]): list of patterns found
    """
    #print('Empresa: ', company_name)
    numero_ventanas = 0
    aceptados_findCommon_por_patron = {}
    aceptados_findCommon_por_patron['total'] = 0
    for keys in patterns_dictionary.keys():
        if keys == 'rest_normalized':
            continue
        aceptados_findCommon_por_patron[keys] = 0
        aceptados_findCommon_por_patron[f'{keys}_distancia_findCommon'] = 0


    lista_ventanas = []

    patterns_found = []
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
            if key == 'rest_normalized':
                continue
            temp_dict = {key: patterns_dictionary[key]}
            i = 0
            while i < len(company_data) - window_width - 1:
                right_window_index = i + window_width
                if right_window_index >= len(company_data):
                    break
                sliced_dataframe = company_data.iloc[i:right_window_index]
                normalized_vector = normalizeVector(sliced_dataframe['SMA'].tolist())
                new_pattern_type, best_distance_found = findCommonPattern(normalized_vector, temp_dict)
                lista_ventanas.append((sliced_dataframe, new_pattern_type, best_distance_found, numero_ventanas))
                if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < 40:
                    aceptados_findCommon_por_patron['total'] += 1 # <-----------------------------------
                    aceptados_findCommon_por_patron[new_pattern_type] += 1 # <-----------------------------------
                    aceptados_findCommon_por_patron[f'{new_pattern_type}_distancia_findCommon'] += best_distance_found # <-----------------------------------
                    left_index, right_index = MIenhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), temp_dict, DIVISIONS, indice_tabla = -1)
                    dataframe_segment = sliced_dataframe[left_index:right_index] #Esto sin ventana mejorada
                    longer_dataframe = company_data[i + left_index:] #Quitar left_index si no se usa enhanced dataframe
                    pattern_tendency = findPatternTendency(dataframe_segment, longer_dataframe, new_pattern_type)
                    if pattern_tendency != None:
                        new_pattern = Pattern(new_pattern_type, pattern_tendency[1], company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], best_distance_found, len(dataframe_segment.index) ,pattern_tendency[2])
                        patterns_found.append(new_pattern)
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
            new_pattern_type, best_distance_found = findCommonPattern(normalized_vector, patterns_dictionary)
            lista_ventanas.append((sliced_dataframe, new_pattern_type, best_distance_found, numero_ventanas))
            if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < 40:
                aceptados_findCommon_por_patron['total'] += 1 # <-----------------------------------
                aceptados_findCommon_por_patron[new_pattern_type] += 1 # <-----------------------------------
                aceptados_findCommon_por_patron[f'{new_pattern_type}_distancia_findCommon'] += best_distance_found # <-----------------------------------
                left_index, right_index = MIenhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), patterns_dictionary, DIVISIONS, indice_tabla)
                dataframe_segment = sliced_dataframe[left_index:right_index] #Esto sin ventana mejorada
                longer_dataframe = company_data[i + left_index:] #Quitar left_index si no se usa enhanced dataframe
                pattern_tendency = findPatternTendency(dataframe_segment, longer_dataframe, new_pattern_type)
                if pattern_tendency != None:
                    new_pattern = Pattern(new_pattern_type, pattern_tendency[1], company_name, str(dataframe_segment.iloc[0].name), str(dataframe_segment.iloc[len(dataframe_segment) - 1].name), pattern_tendency[0], best_distance_found, len(dataframe_segment.index), pattern_tendency[2])
                    patterns_found.append(new_pattern)
                i += right_index
            else:
                i += INCREMENT
            numero_ventanas += 1
    #print(f'Para la empresa {company_name} se exploraron {numero_ventanas} originales, de ellas se aceptaron {contador_aceptadas_por_findCommon} por findCommon')
    '''
    output_dir = '../collected_data/ventanas_originales/'
    os.makedirs(output_dir, exist_ok=True)
    dpi = 100
    figsize_x = 900 / dpi
    figsize_y = 500 / dpi
    for ventana in lista_ventanas:
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
        plt.plot(ventana[0]['SMA'].tolist())
        ax.plot(ventana[0]['SMA'].tolist())
        # Title to plot
        ax.set_title(f'{str(ventana[0].iloc[0].name)} {str(ventana[0].iloc[len(ventana[0]) - 1].name)}')
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
       # save the image with the plot
        img = Image.open(buf)
        img.save(f'{output_dir}{company_name}_{ventana[3]}_{ventana[1]}_{round(ventana[2], 1)}.png') # numero de ventana, tipo de patron, distancia
        buf.close()
        plt.close(fig)
    '''
    for key, value in aceptados_findCommon_por_patron.items():
        if key.endswith('_distancia_findCommon'):
            if value != 0:
                aceptados_findCommon_por_patron[key] = value / aceptados_findCommon_por_patron[key[:-21]]
    #print(aceptados_findCommon_por_patron)
    return patterns_found, numero_ventanas, aceptados_findCommon_por_patron


empresas = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA', 'BBVA', 'JPM', 'SAN', 'BLK', 'V']
divisiones = [[1, 2, 3], [1,2,3,4], [1,2,3,4,5], [1,2,3,4,5,6]]
tamanios_ventana = [100, 120, 140, 160, 180, 200, 220, 240, 260]
fechas = [(date(2023, 1, 2), date(2024, 1, 2)), # 1 año 0
          (date(2022, 1, 2), date(2023, 6, 1)), # 1 año y medio 1
          (date(2022, 1, 2), date(2024, 1, 2)), # 2 años 2
          (date(2020, 1, 2), date(2024, 1, 2)), # 4 años 4
          (date(2019, 1, 2), date(2024, 1, 2)), # 5 años 5
          (date(2018, 1, 2), date(2024, 1, 2)), # 6 años 6
          (date(2016, 1, 2), date(2024, 1, 2)), # 8 años 8
          (date(2015, 1, 2), date(2024, 1, 2)), # 9 años 9
          (date(2014, 1, 2), date(2024, 1, 2)), # 10 años 10
          (date(2013, 1, 2), date(2024, 1, 2)), # 11 años 11
          (date(2011, 1, 2), date(2024, 1, 2)), # 13 años 12
          (date(2009, 1, 2), date(2024, 1, 2)),] # 15 años 13



fecha_numerodepuntos = {
  0: 250,
  1: 354,
  2: 501,
  3: 1006,
  4: 1258,
  5: 1509,
  6: 2012,
  7: 2264,
  8: 2516,
  9: 2768,
  10: 3270,
  11: 3774
}

patrones_a_estudiar = ['double_top', 'double_bottom', 'head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle',
                       'descending_triangle']
numero_patrones_a_cargar = [15]


if __name__ == '__main__':
  # wait for enter input
  input('Press enter to start the process')
  datos_tabla_general = []
  datos_tabla_por_patron = []
  indice_tabla = 0
  for NUMERO in numero_patrones_a_cargar:
    diccionario_patrones = loadPatterns(NUMERO, patrones_a_estudiar)
    for FECHA in fechas:
      for TAMANO in tamanios_ventana:
        for DIVISION in divisiones:
          patrones_encontrados = []
          total_ventanas_exploradas = 0
          aceptados_findCommon = {}
          aceptados_findCommon['total'] = 0
          for keys in patrones_a_estudiar:
            aceptados_findCommon[keys] = 0
            aceptados_findCommon[f'{keys}_distancia_findCommon'] = 0
          print(f'Explorando combinación: {NUMERO} patrones, {FECHA} ({fecha_numerodepuntos[fechas.index(FECHA)]}), tamaño ventana: {TAMANO}, num_divisiones: {len(DIVISION)}')
          start_time = time.time()
          for EMPRESA in empresas:
            dataframe = get_company_data.getCompanyDataWithYahoo(EMPRESA, FECHA[0].strftime("%Y-%m-%d"), FECHA[1].strftime("%Y-%m-%d"))
            if dataframe is None or dataframe.empty or len(dataframe.index) < fecha_numerodepuntos[fechas.index(FECHA)]:
              print(f'No se ha podido descargar los datos de la empresa {EMPRESA} en el rango de fechas {FECHA[0]} - {FECHA[1]}')
              continue
            encontrados_empresa, ventanas_exp, c_aceptados_findCommon = MIfindHistoricPatterns(TAMANO, dataframe, 
                                                                    diccionario_patrones, EMPRESA, DIVISION, indice_tabla)
            patrones_encontrados = patrones_encontrados + encontrados_empresa
            if ventanas_exp == 0:
                print('Error en combinación', NUMERO, FECHA, TAMANO, DIVISION, EMPRESA)
                continue
            total_ventanas_exploradas += ventanas_exp
            for key, value in c_aceptados_findCommon.items():
              if key.endswith('_distancia_findCommon'):
                if aceptados_findCommon[key] != 0:
                  aceptados_findCommon[key] = (c_aceptados_findCommon[key] + aceptados_findCommon[key]) / 2
                else:
                    aceptados_findCommon[key] = c_aceptados_findCommon[key]
              else:
                aceptados_findCommon[key] += value
          end_time = time.time()
          if total_ventanas_exploradas == 0:
            continue
          datos_tabla_general.append({'indice': indice_tabla, 'num_patrones_cargar': NUMERO, 'rango_fecha': calcular_tiempo(FECHA), 'numero_puntos': fecha_numerodepuntos[fechas.index(FECHA)],
                                      'tamano_ventana': TAMANO, 
                                      'divisiones': DIVISION, 'numero_ventanas_exploradas': total_ventanas_exploradas, 
                                      'total_aceptados_findCommon': aceptados_findCommon['total'], 
                                      'numero_encontrados_total': len(patrones_encontrados), 
                                      'porcentaje_aceptacion_findCommon': round(aceptados_findCommon['total'] / total_ventanas_exploradas * 100, 1), 
                                      'porcentaje_aceptacion_tendency': round(len(patrones_encontrados) / aceptados_findCommon['total'] * 100, 1), 
                                      'tiempo': round(end_time - start_time, 2)})
          condicion = lambda patron, tipo: patron.pattern_type == tipo 
          for tipo_patron in patrones_a_estudiar:
            total_encontrados = sum(1 for patron in patrones_encontrados if condicion(patron, tipo_patron))
            if total_encontrados == 0:
              datos_tabla_por_patron.append({'indice': indice_tabla, 'tipo_patron': tipo_patron, 'numero_encontrados': total_encontrados, 
                                            'distancia_promedio': 0, 
                                            'distancia_promedio_findCommon': round(aceptados_findCommon[f'{tipo_patron}_distancia_findCommon'], 2), 
                                            'num_aceptados_findCommon_patron': round(aceptados_findCommon[tipo_patron], 2), 'tamaño_medio': 0,
                                            'porcentaje_aceptacion_tendency': 0})
            else:
              datos_tabla_por_patron.append({'indice': indice_tabla, 'tipo_patron': tipo_patron, 'numero_encontrados': total_encontrados, 
                    'distancia_promedio': round(np.mean([patron.distance for patron in patrones_encontrados if condicion(patron, tipo_patron)]), 2), 
                    'distancia_promedio_findCommon': round(aceptados_findCommon[f'{tipo_patron}_distancia_findCommon'], 2), 
                    'num_aceptados_findCommon_patron': aceptados_findCommon[tipo_patron], 
                    'tamaño_medio': round(np.mean([patron.length_pattern for patron in patrones_encontrados if condicion(patron, tipo_patron)]), 2),
                    'porcentaje_aceptacion_tendency': round(total_encontrados / len(patrones_encontrados) * 100, 1)})
          indice_tabla += 1
  df_tabla_general = pd.DataFrame(datos_tabla_general)
  df_tabla_por_patron = pd.DataFrame(datos_tabla_por_patron)
  df_tabla_general.to_csv('./tabla_general_15_v2.csv', index=False)
  df_tabla_por_patron.to_csv('./tabla_por_patron_15_v2.csv', index=False)
        