from torchvision import transforms
import torch.nn as nn
import torch

from datetime import date, timedelta
import get_company_data
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
import numpy as np
from PIL import Image
import io
import time
import random

import pattern_utils
import Pattern as p
import tendencyCalculator as tc
from random import randint
import normalize_utils as nu

from datetime import datetime, timedelta

from network_utils import transform, CNN_all, CNN
from prueba_red import getIncrement, applySimpleMovingAverage, convertir_patron_a_imagen, PatternToImage

pattern_index_one_by_one = {
    'double_top': 0,
    'double_bottom': 1,
    'ascending_triangle': 2,
    'descending_triangle': 3,
    'head_and_shoulders': 4,
    'inv_head_and_shoulders': 5
}

index_pattern_all = {
    0: 'ascending_triangle',
    1: 'descending_triangle',
    2: 'double_bottom',
    3: 'double_top',
    4: 'head_and_shoulders',
    5: 'inv_head_and_shoulders'
}

index_pattern_one_by_one = {
    0: 'double_top',
    1: 'double_bottom',
    2: 'ascending_triangle',
    3: 'descending_triangle',
    4: 'head_and_shoulders',
    5: 'inv_head_and_shoulders'
}

def classify_all(image_tensor, model):
    predicted_final = -1
    confidence_final = -1
    model.eval() 
    with torch.no_grad():
        outputs = model(image_tensor)
        confidence, predicted = torch.max(outputs.data, 1)
        predicted = predicted.item()
        confidence = confidence.item()
        if predicted < 6:
            confidence_final = confidence
            predicted_final = predicted
    return predicted_final, confidence_final

def classify_one_by_one(image_tensor, models):
    predicted_final = -1
    confidence_final = -1
    for pattern, model in models.items():
        model.eval() 
        with torch.no_grad():
            outputs = model(image_tensor)
            confidence, predicted = torch.max(outputs.data, 1)
            if predicted == 0:
                confidence_final = confidence.item()
                predicted_final = pattern_index_one_by_one[pattern]
                break
    return predicted_final, confidence_final


def trainHistoricDatabase(company, initial_date, final_date, models,  one_by_one, window_size = 120):
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, initial_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print('empty or None dataframe x2')
        return []
    #WINDOW_SIZE, DIVISIONS = pattern_utils.windowSizeAndDivisions(company_dataframe)
    #print(WINDOW_SIZE, DIVISIONS)
    patterns_found = MIfindHistoricPatterns_red(window_size, company_dataframe, models, company, one_by_one)
    #for pattern in patterns_found:
        #fig = PatternToImage(pattern)
        #fig.savefig(f'imagenes_patrones_detectados/{pattern.pattern_type}/{pattern.pattern_type}_{randint(0, 1000)}.png')
    return patterns_found


def MIfindHistoricPatterns_red(window_width, company_data, models, company_name, one_by_one):
    patterns_found_p = []
    i = 0
    increment = getIncrement(window_width)
    #separated_execute = False PARA LA RED NEURONAL POR AHORA VOY A IGNORAR EL SEPARATED_EXECUTE
    company_data = applySimpleMovingAverage(company_data) 
    #if ('double_top' in patterns_dictionary.keys() and 'head_and_shoulders' in patterns_dictionary.keys()) or ('double_bottom' in patterns_dictionary.keys() and 'inv_head_and_shoulders' in patterns_dictionary.keys()):
        #separated_execute = True
    while i < len(company_data) - window_width - 1:
        right_window_index = i + window_width
        if right_window_index >= len(company_data):
            break
        sliced_dataframe = company_data.iloc[i:right_window_index]
        normalized_vector = nu.normalizeVector(sliced_dataframe['SMA'].tolist()) 
        image = convertir_patron_a_imagen(normalized_vector)
        #image.save(f'imagenes_patrones_detectados/z_prueba/{company_name}_{i}.png')
        image_tensor = transform(image)
        if one_by_one:
            predicted, confidence = classify_one_by_one(image_tensor, models)
        else:
            predicted, confidence = classify_all(image_tensor, models)
        if predicted >= 0 and confidence >= 5.0:
            if one_by_one:
                pattern_type = index_pattern_one_by_one[predicted]
            else:
              pattern_type = index_pattern_all[predicted]
            # aquí debería hacer enhanceDataframeDistancesMean para que me de left_index y right_index pero por ahora lo dejo así para probar si funciona y me lo ahorro
            pattern_tendency = tc.findPatternTendency(sliced_dataframe, sliced_dataframe, pattern_type)
            if pattern_tendency != None:
                new_pattern_Pattern = p.Pattern(pattern_type, pattern_tendency[1], company_name, str(sliced_dataframe.iloc[0].name), str(sliced_dataframe.iloc[len(sliced_dataframe) - 1].name), pattern_tendency[0], 40, pattern_tendency[2])
                patterns_found_p.append(new_pattern_Pattern)
            i += int(round(window_width * 0.5, 0)) # COMO TODAVÍA NO ESTOY HACIENDO LOS PATRONES CON tendency DEJARLO ASÍ PERO CUANDO TENGA LO OTRO AUMENTAR HASTA DONDE ACABA EL PATRÓN
        else:
            i += increment
    return patterns_found_p


conjunto_empresas = [
   [
    'EVRG', 'EVRI', 'EVTC', 'EVTL', 'EVTV', 'EW', 'EWBC', 'EWCZ', 'EWTX', 'EXAI',
    'EXAS', 'EXC', 'EXEL', 'EXFY', 'EXK', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPI',
    'EXPO', 'EXR', 'EXTO', 'EXTR', 'EYE', 'EYEN', 'EYPT', 'EZFL', 'EZGO', 'EZPW',
    'F', 'FA', 'FAAS', 'FAF', 'FAMI', 'FANG', 'FANH', 'FARM', 'FARO', 'FAST',
    'FAT', 'FATBB', 'FATE', 'FATH', 'FBIN', 'FBIO', 'FBIZ', 'FBK', 'FBLG', 'FBMS'
   ],
   [
    'GBDC', 'GBIO', 'GBLI', 'GBNY', 'GBR', 'GBTG', 'GBX', 'GCBC', 'GCI', 'GCMG',
    'GCO', 'GCT', 'GCTK', 'GCTS', 'GD', 'GDC', 'GDDY', 'GDEN', 'GDEV', 'GDHG',
    'GDOT', 'GDRX', 'GDS', 'GDST', 'GDTC', 'GDYN', 'GE', 'GECC', 'GEF', 'GEF.B',
    'GEG', 'GEHC', 'GEL', 'GEN', 'GENC', 'GENE', 'GENI', 'GENK', 'GEO', 'GEOS',
    'GERN', 'GES', 'GETR', 'GETY', 'GEV', 'GEVO', 'GFAI', 'GFF', 'GFI', 'GFL',
    'GFR'
   ],
   [
    'AMTX', 'AMWD', 'AMWL', 'AMX', 'AMZN', 'AN', 'ANAB', 'ANDE', 'ANEB', 'ANET',
    'ANF', 'ANGH', 'ANGI', 'ANGO', 'ANIK', 'ANIP', 'ANIX', 'ANL', 'ANNX', 'ANRO',
    'ANSC', 'ANSS', 'ANTE', 'ANTX', 'ANVS', 'ANY', 'AOGO', 'AOMR', 'AON', 'AONC',
    'AORT', 'AOS', 'AOSL', 'AOUT', 'AP', 'APA', 'APAM', 'APCA', 'APCX', 'APD',
    'APDN', 'APEI', 'APG', 'APGE', 'APH', 'API', 'APLD', 'APLE', 'APLM', 'APLS'
   ],
   [
    'AXR', 'AXS', 'AXSM', 'AXTA', 'AXTI', 'AY', 'AYI', 'AYRO', 'AYTU', 'AZ', 'AZEK',
    'AZN', 'AZO', 'AZPN', 'AZTA', 'AZTR', 'AZUL', 'AZZ', 'DGHI', 'DGICA', 'DGICB',
    'DGII', 'DGLY', 'DGX', 'DH', 'DHAC', 'DHAI', 'DHC', 'DHI', 'DHIL', 'DHR', 'DHT',
    'DHX', 'DIBS', 'DIN', 'DINO', 'DIOD', 'DIS', 'DIST', 'DIT', 'DJCO', 'DJT', 'DK',
    'DKL', 'DKNG', 'DKS', 'DLA', 'DLB', 'DLHC', 'DLNG', 'DLO'
   ]
]


if __name__ == "__main__":
    model_double_top = CNN()
    model_double_bottom = CNN()
    model_ascending_triangle = CNN()
    model_descending_triangle = CNN()
    model_head_and_shoulders = CNN()
    model_inv_head_and_shoulders = CNN()
    model_double_top.load_state_dict(torch.load('double_top_model.pth'))
    model_double_bottom.load_state_dict(torch.load('double_bottom_model.pth'))
    model_ascending_triangle.load_state_dict(torch.load('ascending_triangle_model.pth'))
    model_descending_triangle.load_state_dict(torch.load('descending_triangle_model.pth'))
    model_head_and_shoulders.load_state_dict(torch.load('head_and_shoulders_model.pth'))
    model_inv_head_and_shoulders.load_state_dict(torch.load('inv_head_and_shoulders_model.pth'))
    models = {
        'double_top': model_double_top,
        'double_bottom': model_double_bottom,
        'ascending_triangle': model_ascending_triangle,
        'descending_triangle': model_descending_triangle,
        'head_and_shoulders': model_head_and_shoulders,
        'inv_head_and_shoulders': model_inv_head_and_shoulders
    }

    model = CNN_all()
    model.load_state_dict(torch.load('all_patterns_model.pth'))
    input('Press enter to start')
    for conjunto in conjunto_empresas:
        tiempo_total_separado = 0
        tiempo_total_junto = 0
        encontrados_juntos = []
        encontrados_separados = []
        for empresa in conjunto:
            tiempo_inicial_separado = time.time()
            encontrados_separados = encontrados_separados + trainHistoricDatabase(empresa, datetime(2018, 1, 1), datetime(2024, 1, 1), models, True, 120)
            tiempo_final_separado = time.time()
            tiempo_total_separado += tiempo_final_separado - tiempo_inicial_separado
            tiempo_inicial_junto = time.time()
            encontrados_juntos = encontrados_juntos + trainHistoricDatabase(empresa, datetime(2018, 1, 1), datetime(2024, 1, 1), model, False, 120)
            tiempo_final_junto = time.time()
            tiempo_total_junto += tiempo_final_junto - tiempo_inicial_junto
        print(f'conjunto: {conjunto_empresas.index(conjunto) + 1}')
        print(f'Tiempo total separado: {tiempo_total_separado}')
        print(f'Tiempo total junto: {tiempo_total_junto}')
        print(f'Total encontrados separado: {len(encontrados_separados)}')
        print(f'Total encontrados junto: {len(encontrados_juntos)}')
        print('---------------------------------------')
        for pattern in encontrados_separados:
            fig = PatternToImage(pattern)
            fig.savefig(f'imagenes_patrones_detectados/z_detectados_uno_por_uno/{pattern.pattern_type}_{randint(0, 1000)}.png')    
        for pattern in encontrados_juntos:
            fig = PatternToImage(pattern)
            fig.savefig(f'imagenes_patrones_detectados/z_detectados_juntos/{pattern.pattern_type}_{randint(0, 1000)}.png')