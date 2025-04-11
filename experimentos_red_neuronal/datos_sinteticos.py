import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
import random
import os
import csv
from PIL import Image
import io

PATTERNS_FILE_PATH = '../patterns/'

patrones_a_elegir = {
  'double_top': ['AVA110', 'BMO59', 'CTBI750', 'FJAN125', 'ACCO906'],   
  'double_bottom': ['ADC115', 'ADP866', 'CQQQ314', 'SKX942', 'WHR646'],
  'head_and_shoulders': ['AAT481', 'ADP741', 'ALB954', 'BANR384', 'EZPW536'],
  'inv_head_and_shoulders': ['AA192', 'ACIW37', 'UAE768', 'BBW938', 'ARWR514'],
  'ascending_triangle': ['BALL737', 'BANX186', 'BBCA718', 'ECH247', 'EWC392'],
  'descending_triangle': ['AAP554', 'CHT902', 'LOV800', 'RRC785', 'TTT763']
}

def normalizeVector(vector):
    """Normalize a given vector with max min normalization"""
    max_number = 0
    min_number = 99999999
    for number in vector:
        number = float(number)
        if number > max_number:
            max_number = number
        if number < min_number:
            min_number = number
    
    normalized_vector = []

    if max_number == min_number:
        #print(vector)
        return [1, 1, 1] # Devuelve un vector que nunca va a ser escogido por el algoritmo de búsqueda de patrones

    for number in vector:
        number = float(number)
        #print(vector)
        normalized_number = (number - min_number) / (max_number - min_number)
        normalized_vector.append(round(normalized_number, 3))

    return normalized_vector

def calculateArraySimpleMovingAverage(array, window_size):
    """Calculate the simple moving average for a given array and window size

        Args:  
            array (List[]): array containing the prices
            window_size (int): size of the window to calculate the moving average  
        Return:  
            array (List[]): array containing the prices and the moving average
    """
    results = []
    for i in range(len(array)):
        if i >= window_size:
            aux = sum(array[i-window_size:i]) / window_size
            results.append(round(aux,4))
    return results


def obtener_patrones(patrones_a_elegir=None):
    '''
    patrones_a_elegir: diccionario con el tipo de patron y los patrones que se quieren cargar,
    si esta a None se cargan todos los patrones
    '''
    patrones = {
      'double_top': [],
      'double_bottom': [],
      'head_and_shoulders': [],
      'inv_head_and_shoulders': [],
      'ascending_triangle': [],
      'descending_triangle': []
    }
    if patrones_a_elegir is None:
        for tipo_patron in patrones.keys():
            ruta = '../collected_data_csv/' + tipo_patron + '/'
            file_list = os.listdir(ruta)
            for file in file_list:
                single_file_results = []
                with open(ruta + file) as csvfile:
                    print(file)
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        single_file_results.append(round(float(row[1]), 3))
                patrones[tipo_patron].append((single_file_results, file))
    else:
        for tipo_patron in patrones_a_elegir.keys():
            for nombre_fichero_patron in patrones_a_elegir[tipo_patron]:
                single_file_results = []
                with open(PATTERNS_FILE_PATH + tipo_patron + '/' + nombre_fichero_patron + '.csv') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        single_file_results.append(round(float(row[1]), 3))
                patrones[tipo_patron].append(single_file_results)
    return patrones

def crear_datos_sinteticos(patron, numero_datos_sinteticos, tipo_patron):
    '''
    patron: lista con los datos del patron
    numero_datos_sinteticos: numero de datos sinteticos que se quieren generar
    '''
    patrones_creados = []
    porcentaje_minimo_puntos_modificar = 0.40
    porcentaje_maximo_puntos_modificar = 0.70
    porcentaje_minimo_puntos_anadir = 0.30
    porcentaje_maximo_puntos_anadir = 0.50
    if len(patron) < 30:
        porcentaje_minimo_puntos_anadir = 0.10
    if tipo_patron == 'double_top' or tipo_patron == 'double_bottom':
        porcentaje_maximo_puntos_modificar = 0.30
        porcentaje_maximo_puntos_anadir = 0.30
    for i in range(numero_datos_sinteticos):
        nuevo_patron = patron.copy()
        longitud_patron = len(nuevo_patron)
        num_puntos_a_anadir = random.randint(round(longitud_patron*porcentaje_minimo_puntos_anadir), round(longitud_patron*porcentaje_maximo_puntos_anadir))
        for i in range(num_puntos_a_anadir):
            indice = random.randint(0, longitud_patron - 2)
            # inserta un punto en la posicion indice + 1 que sea la media entre el punto en la posicion indice y el punto en la posicion indice + 1
            nuevo_patron.insert(indice + 1, (nuevo_patron[indice] + nuevo_patron[indice + 1]) / 2)
            # inserta un punto en la posicion indice + 1 usando la interpolacion cubica
            #nuevo_patron.insert(indice + 1, cs(indice + 1))
        longitud_patron = len(nuevo_patron)
        num_puntos_a_reducir = random.randint(round(longitud_patron*porcentaje_minimo_puntos_modificar), round(longitud_patron*porcentaje_maximo_puntos_modificar))
        num_puntos_a_aumentar = random.randint(round(longitud_patron*porcentaje_minimo_puntos_modificar), round(longitud_patron*porcentaje_maximo_puntos_modificar))
        indices_puntos_a_reducir = random.sample(range(1, longitud_patron - 2), num_puntos_a_reducir)
        indices_puntos_a_aumentar = random.sample(range(1, longitud_patron - 2), num_puntos_a_aumentar)
        for indice in indices_puntos_a_reducir:
            # calcular aleatoriamente un valor entre el 0.1 y el 0.3 de valor
            sustraendo = random.uniform(0.1, 0.5) * nuevo_patron[indice]
            nuevo_patron[indice] = nuevo_patron[indice] - sustraendo
        for indice in indices_puntos_a_aumentar:
            # calcular aleatoriamente un valor entre el 0.1 y el 0.3 de valor
            sumando = random.uniform(0.1, 0.5) * nuevo_patron[indice]
            nuevo_patron[indice] = nuevo_patron[indice] + sumando
        #x = np.arange(longitud_patron)
        #cs = CubicSpline(x, nuevo_patron)
        patrones_creados.append(nuevo_patron)
    return patrones_creados

def mostrar_patron(patron):
    plt.plot(patron)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of the Data')
    plt.grid(True)
    plt.show()

def guardarImagenPatron(patron, tipo_patron, patron_original):
    if not os.path.exists('./experimento_datos_sinteticos/' + tipo_patron):
        os.makedirs('./experimento_datos_sinteticos/' + tipo_patron)
    output_dir = './experimento_datos_sinteticos/' + tipo_patron
    dpi = 100
    figsize_x = 256 / dpi
    figsize_y = 64 / dpi
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
    ax.plot(patron, color='black', linewidth=1)
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
    numero = random.randint(1, 10000)
    numero2 = random.randint(1, 10000)
    numero3 = random.randint(1, 10000)
    numero4 = random.randint(1, 10000)
    numero5 = random.randint(1, 10000)
    filename = f'{tipo_patron} - {patron_original} - sintetico - {numero} - {numero2} - {numero3} - {numero4} - {numero5}'
    guardarCSVpatron(patron, filename, output_dir)
    #final_output_dir = os.path.join(output_dir, tipo_patron)
    output_filepath = os.path.join(output_dir, f"{filename}.png")
    # Save the image to the specified directory
    image.save(output_filepath)
    buf.close()
    plt.close(fig)

def guardarCSVpatron(patron, nombre, final_output_dir):
    output_filepath = os.path.join(final_output_dir, f"{nombre}.csv")
    #normalize the data
    normalized_data = normalizeVector(patron)
    normalized_dataframe = pd.DataFrame(normalized_data)
    normalized_dataframe.to_csv(output_filepath, sep=',')

SINTETICOS_POR_PATRON = {
    'double_top': 0,
    'double_bottom': 0,
    'head_and_shoulders': 0,
    'inv_head_and_shoulders': 0,
    'ascending_triangle': 47,
    'descending_triangle': 0
}


if __name__ == '__main__':
    patrones_cargados = obtener_patrones()
    for tipo_patron in patrones_cargados.keys():
        print('Tipo patron: ', tipo_patron)
        for patron in patrones_cargados[tipo_patron]:
            #print('Patron original')
            #mostrar_patron(patron)
            patrones_creados = crear_datos_sinteticos(patron[0], SINTETICOS_POR_PATRON[tipo_patron], tipo_patron)
            for patron_creado in patrones_creados:
                if len(patron_creado) > 30:
                  patron_creado = calculateArraySimpleMovingAverage(patron_creado, 3)
                elif len(patron_creado) > 100:
                  patron_creado = calculateArraySimpleMovingAverage(patron_creado, 5)
                elif len(patron_creado) > 200:
                  patron_creado = calculateArraySimpleMovingAverage(patron_creado, 8)
                guardarImagenPatron(patron_creado, tipo_patron, patron[1])
                # Guarda la imagen del patron creado en la ruta ./experimento_datos_sinteticos/tipo_patron/nombre_patron.png'''
                
