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

SMA_VALUE = 3
TAMANIO_MINIMO_DATAFRAME = 15

pattern_index = {
    'double_top': 0,
    'double_bottom': 1,
    'ascending_triangle': 2,
    'descending_triangle': 3,
    'head_and_shoulders': 4,
    'inv_head_and_shoulders': 5
}

index_pattern = {
    0: 'double_top',
    1: 'double_bottom',
    2: 'ascending_triangle',
    3: 'descending_triangle',
    4: 'head_and_shoulders',
    5: 'inv_head_and_shoulders'
}

def PatternToImage(pattern):
    """
    Convierte un objeto Pattern a una imagen
    """
    fig = Figure(figsize = (9,5), dpi = 100)
    plot1 = fig.add_subplot(111)
    plot1.plot(pattern.dataframe_segment.iloc[:, 0])
    df2 = pattern.points
    if pattern.points is not None:
        if (isinstance(pattern.points, list)):
            for x in pattern.points:
                plot1.plot(x)
        else:  
            plot1.plot(df2)
    #fig.suptitle(f'{pattern.company_name} {pattern.pattern_type} {pattern.starting_date[:10]} - {pattern.ending_date[:10]} Distance: {round(pattern.distance, 2)}')
    fig.suptitle(f'{pattern.company_name} {pattern.pattern_type} {pattern.starting_date[:10]} - {pattern.ending_date[:10]}')
    return fig



def getIncrement(window_width):
    return int(round(window_width * 0.25, 0))


def applySimpleMovingAverage(company_dataframe):
  if (company_dataframe.shape[0] > 30):
    company_dataframe = pattern_utils.calculateSimpleMovingAverage(company_dataframe, 3)
    company_dataframe = company_dataframe.iloc[SMA_VALUE-1:]
  elif (company_dataframe.shape[0] > 100):
    company_dataframe = pattern_utils.calculateSimpleMovingAverage(company_dataframe, 5)
    company_dataframe = company_dataframe.iloc[SMA_VALUE-1:]
  elif (company_dataframe.shape[0] > 200):
    company_dataframe = pattern_utils.calculateSimpleMovingAverage(company_dataframe, 8)
    company_dataframe = company_dataframe.iloc[SMA_VALUE-1:]
  else:
    company_dataframe = pattern_utils.calculateSimpleMovingAverage(company_dataframe, 1) # No le aplica la media movil  
  return company_dataframe

def classify(image_tensor, models):
    predicted_final = -1
    confidence_final = -1
    for pattern, model in models.items():
        model.eval() 
        with torch.no_grad():
            outputs = model(image_tensor)
            confidence, predicted = torch.max(outputs.data, 1)
            if predicted == 0:
                confidence_final = confidence.item()
                predicted_final = pattern_index[pattern]
                break
    return predicted_final, confidence_final

def convertir_patron_a_imagen(patron):
    dpi = 100
    figsize_x = 256 / dpi
    figsize_y = 64 / dpi
    fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(patron, color='black', linewidth=1)
    ax.axis('off')
    fig.patch.set_visible(False)
    fig.tight_layout(pad=0)
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    # Load this image into PIL Image object and convert to black and white
    # image = Image.open(buf).convert('L')
    image = Image.open(buf).convert('RGB')
    plt.close(fig)
    buf.close()
    return image
# ------------------------------------PATRONES HISTÓRICOS----------------------------------------
def trainHistoricDatabase(company, initial_date, final_date, models, window_size = 120):
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company, initial_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))
    if company_dataframe is None or company_dataframe.empty:
        print('empty or None dataframe x2')
        return []
    #WINDOW_SIZE, DIVISIONS = pattern_utils.windowSizeAndDivisions(company_dataframe)
    #print(WINDOW_SIZE, DIVISIONS)
    patterns_found = MIfindHistoricPatterns_red(window_size, company_dataframe, models, company)
    for pattern in patterns_found:
        fig = PatternToImage(pattern)
        fig.savefig(f'imagenes_patrones_detectados/{pattern.pattern_type}/{pattern.pattern_type}_{randint(0, 1000)}.png')
    return patterns_found


def MIfindHistoricPatterns_red(window_width, company_data, models, company_name):
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
        predicted, confidence = classify(image_tensor, models)
        if predicted >= 0 and confidence >= 5.0:
            pattern_type = index_pattern[predicted]
            # aquí debería hacer enhanceDataframeDistancesMean para que me de left_index y right_index pero por ahora lo dejo así para probar si funciona y me lo ahorro
            pattern_tendency = tc.findPatternTendency(sliced_dataframe, sliced_dataframe, pattern_type)
            if pattern_tendency != None:
                new_pattern_Pattern = p.Pattern(pattern_type, pattern_tendency[1], company_name, str(sliced_dataframe.iloc[0].name), str(sliced_dataframe.iloc[len(sliced_dataframe) - 1].name), pattern_tendency[0], 40, pattern_tendency[2])
                patterns_found_p.append(new_pattern_Pattern)
            i += int(round(window_width * 0.5, 0)) # COMO TODAVÍA NO ESTOY HACIENDO LOS PATRONES CON tendency DEJARLO ASÍ PERO CUANDO TENGA LO OTRO AUMENTAR HASTA DONDE ACABA EL PATRÓN
        else:
            i += increment
    return patterns_found_p

# ------------------------------------FIN PATRONES HISTÓRICOS----------------------------------------



# ------------------------------------PATRONES ACTUALES----------------------------------------

def MIfindCurrentPatterns_automatico(company_name, models):
  # Explorar varios tamaños de ventana que sean los que más patrones encuentran, para ello analizar datos experimentos
  # Explorar: 40, 140, 220
  windows_to_explore = [40, 60, 80, 100, 120]
  found_patterns = []
  for window_size in windows_to_explore:
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company_name, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
    pattern_found = MIfindCurrentPatterns_configurable(company_dataframe, models, company_name)
    if pattern_found is not None:
      found_patterns.append(pattern_found)
  for pattern in found_patterns:
    numero = random.randint(0, 1000)
    fig = PatternToImage(pattern)
    fig.savefig(f'imagenes_patrones_detectados/z_prueba/{pattern.pattern_type}_{numero}.png')
  return found_patterns

def MIfindCurrentPatterns_configurable(company_dataframe, models, company_name, attempt=3, failure_increment=0.3, 
                                                                                            acceptance_increment=0.1):
  company_dataframe = applySimpleMovingAverage(company_dataframe) 
  current_best_pattern = None
  pattern_type = None
  best_pattern_Pattern = None
  while True:
    normalized_vector = nu.normalizeVector(company_dataframe['SMA'].tolist())
    image = convertir_patron_a_imagen(normalized_vector)
    image_tensor = transform(image)
    predicted, confidence = classify(image_tensor, models)
    new_pattern = None
    if predicted >= 0 and confidence >= 5.0:
        new_pattern = [normalized_vector, confidence, index_pattern[predicted]]
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
       best_pattern_Pattern = p.Pattern(pattern_type, company_dataframe, company_name, str(company_dataframe.iloc[0].name), str(company_dataframe.iloc[-1].name), None, confidence, pattern_tendency[2])
  return best_pattern_Pattern

# -------------------------------------FIN PATRONES ACTUALES---------------------------------------------------------


# ------------------------------------CÓDIGO RED NEURONAL--------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 clases: con patrón de doble techo o sin él

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
transform = transforms.Compose([
    transforms.Resize((256, 64)),  # Ajustar al tamaño de las imágenes
    transforms.ToTensor()
])

# ------------------------------------FIN CÓDIGO RED NEURONAL---------------------------------------------------------

companies = [
    "B", "BA", "BABA", "BAC", "BACA", "BACK", "BAER", "BAFN", "BAH", "BAK",
    "BALL", "BALY", "BAM", "BANC", "BAND", "BANF", "BANL", "BANR", "BANX",
    "BAOS", "BAP", "BARK", "BASE", "BATL", "BATRA", "BATRK", "BAX", "BAYA",
    "BB", "BBAI", "BBAR", "BBCP", "BBD", "BBDC", "BBDO", "BBGI", "BBIO",
]

if __name__ == '__main__':
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
    for empresa in companies:
        print(f'Analizando empresa {empresa}')
        MIfindCurrentPatterns_automatico(empresa, models)
        #trainHistoricDatabase(empresa, datetime(2021, 1, 2), datetime(2024, 1, 2), models)


