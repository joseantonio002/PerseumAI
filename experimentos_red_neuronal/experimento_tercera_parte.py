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
from experimento_15_v2 import Pattern, calcular_tiempo

from tendencyCalculator import findPatternTendency
from pattern_utils import calculateSimpleMovingAverage
from normalize_utils import normalizeVector
from pattern_utils import findCommonPattern, windowSizeAndDivisions
from dtw_applier import comparePatterns

INCREMENT = 25
SMA_VALUE = 3
MINIMUM_VALUE_TO_APPLY_SMA = 85

# Para rangos de fechas entre 10 meses y 10 años 
# Probar con el código original (solamente cálculos nada de gráficos y pasandole datos por código)
# Probar con 120 y 200 de tamaño de ventana (Los tamaños que se suelen utilizar en el código original)
# Probar código de ventanas y distancias automatizadas (sacar métricas de tiempo, porcentaje de aceptación, número de patrones encontrados, etc.)
# Comparar resultados

# Aclaración: lo de tamaños de ventana y distancias dependiendo del patrón no lo voy a estudiar aquí, 
# se buscarán todos los patrones a la vez

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

def aniadir_datos(datos_finales, tipo_ejecucion, rango_fecha, patrones_encontrados, ventanas_exp, aceptados_findCommon, tiempo_medio):
   tiempo_medio = round(tiempo_medio, 2)
   for pat in patrones_a_estudiar:
      if len(patrones_encontrados) == 0 or (len([patr for patr in patrones_encontrados if patr.pattern_type == pat]) == 0 and pat != 'general'):
         aceptados_findCommon_pat = 0
         media_distancia_findCommon = 0
         if pat == 'general':
           aceptados_findCommon_pat = aceptados_findCommon['total']
           for key in aceptados_findCommon.keys():
              if key.endswith('_distancia_findCommon'):
                media_distancia_findCommon += aceptados_findCommon[key]
           media_distancia_findCommon /= len(patrones_a_estudiar) - 1
         else:
           media_distancia_findCommon = round(aceptados_findCommon[f'{pat}_distancia_findCommon'], 2)
           aceptados_findCommon_pat = aceptados_findCommon[pat]
         datos_finales.append({'Tipo_Ejecucion': tipo_ejecucion, 'Rango_Fecha': rango_fecha, 'Tipo_Patron': pat, 
                                       'numero_ventanas_exploradas': ventanas_exp, 'aceptados_findCommon': aceptados_findCommon_pat, 
                                       'Numero_Encontrados': 0, 'porcentaje_aceptacion_findCommon': round(aceptados_findCommon_pat / ventanas_exp * 100, 1), 
                                       'porcentaje_aceptacion_tendency': 0, 'distancia_promedio': 0, 
                                       'distancia_promedio_findCommon': round(media_distancia_findCommon, 2), 'tamaño_medio': 0, 'Tiempo_Ejecucion': tiempo_medio})
         continue
      if pat == 'general':
         numero_encontrados_patron = len(patrones_encontrados)
         aceptados_findCommon_pat = aceptados_findCommon['total']
         media_distancia_findCommon = 0
         for key in aceptados_findCommon.keys():
           if key.endswith('_distancia_findCommon'):
             media_distancia_findCommon += aceptados_findCommon[key]
         media_distancia_findCommon /= len(patrones_a_estudiar) - 1
         datos_finales.append({'Tipo_Ejecucion': tipo_ejecucion, 'Rango_Fecha': rango_fecha, 'Tipo_Patron': pat, 
                                       'numero_ventanas_exploradas': ventanas_exp, 'aceptados_findCommon': aceptados_findCommon_pat, 
                                       'Numero_Encontrados': numero_encontrados_patron, 
                                       'porcentaje_aceptacion_findCommon':  round(aceptados_findCommon_pat / ventanas_exp * 100, 1), 
                                       'porcentaje_aceptacion_tendency': round(numero_encontrados_patron / aceptados_findCommon_pat * 100, 1), 
                                       'distancia_promedio': round(np.mean([patron.distance for patron in patrones_encontrados]), 2), 
                                       'distancia_promedio_findCommon': round(media_distancia_findCommon, 2), 
                                       'tamaño_medio': round(np.mean([patron.length_pattern for patron in patrones_encontrados]), 2), 
                                       'Tiempo_Ejecucion': tiempo_medio})
      else:
         numero_encontrados_patron = len([patr for patr in patrones_encontrados if patr.pattern_type == pat])
         aceptados_findCommon_pat = aceptados_findCommon[pat]
         condicion = lambda patron, tipo: patron.pattern_type == tipo
         datos_finales.append({'Tipo_Ejecucion': tipo_ejecucion, 'Rango_Fecha': rango_fecha, 'Tipo_Patron': pat, 
                                       'numero_ventanas_exploradas': ventanas_exp, 'aceptados_findCommon': aceptados_findCommon_pat, 
                                       'Numero_Encontrados': numero_encontrados_patron, 
                                       'porcentaje_aceptacion_findCommon':  round(aceptados_findCommon_pat / ventanas_exp * 100, 1), 
                                       'porcentaje_aceptacion_tendency': round(numero_encontrados_patron / aceptados_findCommon_pat * 100, 1), 
                                       'distancia_promedio': round(np.mean([patron.distance for patron in patrones_encontrados if condicion(patron, pat)]), 2), 
                                       'distancia_promedio_findCommon': round(aceptados_findCommon[f'{pat}_distancia_findCommon'], 2), 
                                       'tamaño_medio': round(np.mean([patron.length_pattern for patron in patrones_encontrados if condicion(patron, pat)]), 2), 
                                       'Tiempo_Ejecucion': tiempo_medio})
         
fechas = [(date(2023, 1, 2), date(2024, 1, 2)), # 1 año 
          (date(2022, 1, 2), date(2024, 1, 2)), # 2 años 
          (date(2021, 1, 2), date(2024, 1, 2)), # 3 años
          (date(2020, 1, 2), date(2024, 1, 2)), # 4 años 
          (date(2019, 1, 2), date(2024, 1, 2)), # 5 años 
          (date(2018, 1, 2), date(2024, 1, 2)), # 6 años 
          (date(2016, 1, 2), date(2024, 1, 2)), # 8 años 
          (date(2015, 1, 2), date(2024, 1, 2)), # 9 años 
          (date(2014, 1, 2), date(2024, 1, 2))] # 10 años 

empresas = ['T', 'ABBV', 'ADBE', 'BA', 'CAT', 'CVX', 'CSCO', 'FDX', 'HD', 'MCD']
patrones_a_estudiar = ['double_top', 'double_bottom', 'head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle',
                       'descending_triangle', 'general']

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
                new_pattern_type, best_distance_found = findCommonPattern(normalized_vector, temp_dict)
                lista_ventanas.append((sliced_dataframe, new_pattern_type, best_distance_found, numero_ventanas))
                if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < acceptanceDistanceForPattern(new_pattern_type, original_mode):
                    aceptados_findCommon_por_patron['total'] += 1 # <-----------------------------------
                    aceptados_findCommon_por_patron[new_pattern_type] += 1 # <-----------------------------------
                    aceptados_findCommon_por_patron[f'{new_pattern_type}_distancia_findCommon'] += best_distance_found # <-----------------------------------
                    left_index, right_index = MIenhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), temp_dict, DIVISIONS)
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
            if new_pattern_type != 'rest_normalized' and new_pattern_type != '' and best_distance_found < acceptanceDistanceForPattern(new_pattern_type, original_mode):
                aceptados_findCommon_por_patron['total'] += 1 # <-----------------------------------
                aceptados_findCommon_por_patron[new_pattern_type] += 1 # <-----------------------------------
                aceptados_findCommon_por_patron[f'{new_pattern_type}_distancia_findCommon'] += best_distance_found # <-----------------------------------
                left_index, right_index = MIenhanceDataframeDistancesMean(best_distance_found, new_pattern_type, sliced_dataframe['SMA'].tolist(), patterns_dictionary, DIVISIONS)
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
    for key, value in aceptados_findCommon_por_patron.items():
        if key.endswith('_distancia_findCommon'):
            if value != 0:
                aceptados_findCommon_por_patron[key] = value / aceptados_findCommon_por_patron[key[:-21]]
    return patterns_found, numero_ventanas, aceptados_findCommon_por_patron

if __name__ == '__main__':
  diccionario_patrones = loadPatterns(15, patrones_a_estudiar)
  datos_finales = []
  ventanas_exp_120 = 0
  ventanas_exp_200 = 0
  ventanas_exp_auto = 0
  input('Press enter to start the process')
  for FECHA in fechas:
    print(f'Procesando rango de fechas {FECHA[0].strftime("%Y-%m-%d")} - {FECHA[1].strftime("%Y-%m-%d")}')
    patrones_encontrados_120 = []
    patrones_encontrados_200 = []
    patrones_encontrados_auto = []
    c_aceptados_findCommon_120 = None
    c_aceptados_findCommon_200 = None
    c_aceptados_findCommon_auto = None
    ventanas_exp_120 = 0
    ventanas_exp_200 = 0
    ventanas_exp_auto = 0
    tiempo_medio_120 = 0
    tiempo_medio_200 = 0
    tiempo_medio_auto = 0
    aceptados_findCommon_120 = {}
    aceptados_findCommon_120['total'] = 0
    aceptados_findCommon_200 = {}
    aceptados_findCommon_200['total'] = 0
    aceptados_findCommon_auto = {}
    aceptados_findCommon_auto['total'] = 0
    for keys in patrones_a_estudiar:
      aceptados_findCommon_120[keys] = 0
      aceptados_findCommon_120[f'{keys}_distancia_findCommon'] = 0
      aceptados_findCommon_200[keys] = 0
      aceptados_findCommon_200[f'{keys}_distancia_findCommon'] = 0
      aceptados_findCommon_auto[keys] = 0
      aceptados_findCommon_auto[f'{keys}_distancia_findCommon'] = 0
    for EMPRESA in empresas:
      dataframe_120 = get_company_data.getCompanyDataWithYahoo(EMPRESA, FECHA[0].strftime("%Y-%m-%d"), FECHA[1].strftime("%Y-%m-%d"))
      dataframe_200 = get_company_data.getCompanyDataWithYahoo(EMPRESA, FECHA[0].strftime("%Y-%m-%d"), FECHA[1].strftime("%Y-%m-%d"))
      dataframe_auto = get_company_data.getCompanyDataWithYahoo(EMPRESA, FECHA[0].strftime("%Y-%m-%d"), FECHA[1].strftime("%Y-%m-%d"))
      # Ejecutar Perseum original con 120 ventanas
      start = time.time()
      encontrados_empresa_120, c_ventanas_exp_120, c_aceptados_findCommon_120 = MIfindHistoricPatterns(120, dataframe_120, diccionario_patrones, EMPRESA, [1, 2, 3, 4])
      end = time.time()
      patrones_encontrados_120 = patrones_encontrados_120 + encontrados_empresa_120
      for key, value in c_aceptados_findCommon_120.items():
        if key.endswith('_distancia_findCommon'):
          if aceptados_findCommon_120[key] != 0:
            aceptados_findCommon_120[key] = (c_aceptados_findCommon_120[key] + aceptados_findCommon_120[key]) / 2
          else:
            aceptados_findCommon_120[key] = c_aceptados_findCommon_120[key]
        else:
          aceptados_findCommon_120[key] += value
      tiempo_medio_120 += end - start
      # Ejecutar Perseum original con 200 ventanas
      start = time.time()
      encontrados_empresa_200, c_ventanas_exp_200, c_aceptados_findCommon_200 = MIfindHistoricPatterns(200, dataframe_200, diccionario_patrones, EMPRESA, [1, 2, 3, 4])
      end = time.time()
      patrones_encontrados_200 = patrones_encontrados_200 + encontrados_empresa_200
      for key, value in c_aceptados_findCommon_200.items():
        if key.endswith('_distancia_findCommon'):
          if aceptados_findCommon_200[key] != 0:
            aceptados_findCommon_200[key] = (c_aceptados_findCommon_200[key] + aceptados_findCommon_200[key]) / 2
          else:
            aceptados_findCommon_200[key] = c_aceptados_findCommon_200[key]
        else:
          aceptados_findCommon_200[key] += value
      tiempo_medio_200 += end - start
      # Ejecutar Perseum con ventanas y distancias automatizadas
      start = time.time()
      WINDOW_SIZE, DIVISIONS = windowSizeAndDivisions(dataframe_auto)
      encontrados_empresa_auto, c_ventanas_exp_auto, c_aceptados_findCommon_auto = MIfindHistoricPatterns(WINDOW_SIZE, dataframe_auto, diccionario_patrones, EMPRESA, DIVISIONS, False)
      end = time.time() 
      patrones_encontrados_auto = patrones_encontrados_auto + encontrados_empresa_auto
      for key, value in c_aceptados_findCommon_auto.items():
        if key.endswith('_distancia_findCommon'):
          if aceptados_findCommon_auto[key] != 0:
            aceptados_findCommon_auto[key] = (c_aceptados_findCommon_auto[key] + aceptados_findCommon_auto[key]) / 2
          else:
            aceptados_findCommon_auto[key] = c_aceptados_findCommon_auto[key]
        else:
          aceptados_findCommon_auto[key] += value
      tiempo_medio_auto += end - start
      ventanas_exp_120 += c_ventanas_exp_120
      ventanas_exp_200 += c_ventanas_exp_200
      ventanas_exp_auto += c_ventanas_exp_auto
    tiempo_medio_120 /= len(empresas)
    tiempo_medio_200 /= len(empresas)
    tiempo_medio_auto /= len(empresas)
    aniadir_datos(datos_finales, 'Original_120', calcular_tiempo(FECHA), patrones_encontrados_120, ventanas_exp_120, aceptados_findCommon_120, tiempo_medio_120)
    aniadir_datos(datos_finales, 'Original_200', calcular_tiempo(FECHA), patrones_encontrados_200, ventanas_exp_200, aceptados_findCommon_200, tiempo_medio_200)
    aniadir_datos(datos_finales, 'Auto', calcular_tiempo(FECHA), patrones_encontrados_auto, ventanas_exp_auto, aceptados_findCommon_auto, tiempo_medio_auto)
  df_final = pd.DataFrame(datos_finales)
  df_final.to_csv('resultados_experimento_tercera_parte.csv', index=False)

# Tipo_Ejecucion, Rango_Fecha, Tipo_Patron, numero_ventanas_exploradas, 
# aceptados_findCommon, Numero_Encontrados, porcentaje_aceptacion_findCommon, 
# porcentaje_aceptacion_tendency, distancia_promedio, distancia_promedio_findCommon, 
# tamaño_medio, Tiempo_Ejecucion