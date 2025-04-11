from datetime import date, timedelta

import get_company_data
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from numpy import mean
from pattern_utils import loadPatterns

import main as mn
from prueba_red import PatternToImage
from network_utils import CNN
from torch import load as torch_load
from random import randint


conjunto_empresas = [
    'EXAS', 'EXC', 'EXEL', 'EXFY', 'EXK', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPI',
    'EXPO', 'EXR', 'EXTO', 'EXTR', 'EYE', 'EYEN', 'EYPT', 'EZFL', 'EZGO', 'EZPW',
    'GEG', 'GEHC', 'GEL', 'GEN', 'GENC', 'GENE', 'GENI', 'GENK', 'GEO', 'GEOS'
]

patrones_a_estudiar = ['descending_triangle', 'double_bottom', 'double_top', 'head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle']

found_automatic = {
  'ascending_triangle': 0,
  'descending_triangle': 0,
  'double_bottom': 0,
  'double_top': 0,
  'head_and_shoulders': 0,
  'inv_head_and_shoulders': 0  
}

found_network = {
    'ascending_triangle': 0,
    'descending_triangle': 0,
    'double_bottom': 0,
    'double_top': 0,
    'head_and_shoulders': 0,
    'inv_head_and_shoulders': 0
}

def print_results(number_total_patterns, found, total_time, mode):
    if mode == 'automatic':
        print('Resultados modo automático')
    else:
        print('Resultados modo red')
    print(f'Número total de patrones: {number_total_patterns}')
    print(f'Número total asceding_triangle: {found["ascending_triangle"]}')
    print(f'Número total descending_triangle: {found["descending_triangle"]}')
    print(f'Número total double_bottom: {found["double_bottom"]}')
    print(f'Número total double_top: {found["double_top"]}')
    print(f'Número total head_and_shoulders: {found["head_and_shoulders"]}')
    print(f'Número total inv_head_and_shoulders: {found["inv_head_and_shoulders"]}')
    print(f'Tiempo total: {total_time}')

if __name__ == "__main__":
    patterns_dictionary = loadPatterns(15, patrones_a_estudiar)
    initial_date = date(2018, 1, 2)
    final_date = date(2024, 1, 2)
    model_double_top = CNN()
    model_double_bottom = CNN()
    model_ascending_triangle = CNN()
    model_descending_triangle = CNN()
    model_head_and_shoulders = CNN()
    model_inv_head_and_shoulders = CNN()
    model_double_top.load_state_dict(torch_load('double_top_model.pth'))
    model_double_bottom.load_state_dict(torch_load('double_bottom_model.pth'))
    model_ascending_triangle.load_state_dict(torch_load('ascending_triangle_model.pth'))
    model_descending_triangle.load_state_dict(torch_load('descending_triangle_model.pth'))
    model_head_and_shoulders.load_state_dict(torch_load('head_and_shoulders_model.pth'))
    model_inv_head_and_shoulders.load_state_dict(torch_load('inv_head_and_shoulders_model.pth'))
    models = {
        'double_top': model_double_top,
        'double_bottom': model_double_bottom,
        'ascending_triangle': model_ascending_triangle,
        'descending_triangle': model_descending_triangle,
        'head_and_shoulders': model_head_and_shoulders,
        'inv_head_and_shoulders': model_inv_head_and_shoulders
    }
    network_patterns = []
    automatic_patterns = []
    total_time_network = 0
    total_time_automatic = 0
    for company in conjunto_empresas:
        print(f'Procesando empresa {company}')
        start_network = time.time()
        network_patterns = network_patterns + mn.trainHistoricDatabaseNetwork(company, patterns_dictionary, initial_date, final_date, 180, models)
        #found_network = mn.findCurrentPatternsNetwork(company, models)
        #network_patterns = network_patterns + mn.findCurrentPatternsNetwork(company, models)
        automatic_patterns
        end_network = time.time()
        total_time_network = total_time_network + (end_network - start_network)
        start_automatic = time.time()
        automatic_patterns = automatic_patterns + mn.trainHistoricDatabaseAutomatic(company, patterns_dictionary, initial_date, final_date, 180)
        #automatic_patterns = automatic_patterns + mn.findCurrentPatterns(company, patterns_dictionary, 180)       
        end_automatic = time.time()
        total_time_automatic = total_time_automatic + (end_automatic - start_automatic)
    for pattern in network_patterns:
        found_network[pattern.pattern_type] = found_network[pattern.pattern_type] + 1
        fig = PatternToImage(pattern)
        fig.savefig(f'imagenes_patrones_detectados/z_prueba_moving_avergae/red_{pattern.pattern_type}_{randint(0, 1000)}.png')    
    for pattern in automatic_patterns:
        found_automatic[pattern.pattern_type] = found_automatic[pattern.pattern_type] + 1
        fig = PatternToImage(pattern)
        fig.savefig(f'imagenes_patrones_detectados/z_prueba_moving_avergae/clasico_{pattern.pattern_type}_{randint(0, 1000)}.png')
    print_results(len(network_patterns), found_network, total_time_network, 'red')
    print_results(len(automatic_patterns), found_automatic, total_time_automatic, 'automatic')