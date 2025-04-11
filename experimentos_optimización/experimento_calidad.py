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

from experimento_15_v2 import calcular_tiempo, MIfindHistoricPatterns

empresas = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'BBVA', 'JPM', 'SAN', 'BLK', 'T']

ventanas_pequenias = {
  5: [[1]],
  10: [[1], [1, 2]],
  20: [[1], [1, 2]],
  40: [[1], [1, 2], [1, 2, 3]],
}

ventanas_grandes = {
  80: [[1, 2], [1, 2, 3]],
  90: [[1, 2], [1, 2, 3]],
  100: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  120: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  140: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  160: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  180: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  200: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  220: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  240: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  260: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
  280: [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
}

fechas_ventanas_pequenias = [
          (date(2023, 1, 2), date(2024, 1, 2)), #  1 año
          (date(2021, 1, 2), date(2023, 1, 2)),] # 2 años

fechas_ventanas_grandes = [(date(2023, 1, 2), date(2024, 1, 2)), #  1 año
          (date(2021, 1, 2), date(2023, 1, 2)),
          (date(2017, 1, 2), date(2021, 1, 2))] # 2 años

patrones_a_estudiar = ['double_top', 'double_bottom', 'head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle',
                       'descending_triangle']

if __name__ == "__main__":
    input('Press enter to start the process')
    diccionario_patrones = loadPatterns(15, patrones_a_estudiar)
    for FECHA in fechas_ventanas_pequenias:
        for VENTANA, divisiones in ventanas_pequenias.items():
            dataframe_prueba = get_company_data.getCompanyDataWithYahoo('AAPL', FECHA[0].strftime("%Y-%m-%d"), FECHA[1].strftime("%Y-%m-%d"))
            if len(dataframe_prueba.index) < VENTANA:
                continue
            for DIVISION in divisiones:
                print(f'FECHA: {calcular_tiempo(FECHA)} VENTANA: {VENTANA} DIVISION: {DIVISION}')
                nombre_directorio = f'{VENTANA}_{DIVISION}'
                output_dir = f'./resultados_exp_calidad/{nombre_directorio}/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                patrones_encontrados = []
                for EMPRESA in empresas:
                    dataframe = get_company_data.getCompanyDataWithYahoo(EMPRESA, FECHA[0].strftime("%Y-%m-%d"), FECHA[1].strftime("%Y-%m-%d"))
                    if dataframe is None or dataframe.empty:
                        print(f'No se ha podido descargar los datos de la empresa {EMPRESA} en el rango de fechas {FECHA[0]} - {FECHA[1]}')
                        continue
                    patrones_empresa = MIfindHistoricPatterns(VENTANA, dataframe, diccionario_patrones, EMPRESA, DIVISION)
                    patrones_encontrados = patrones_encontrados + patrones_empresa[0]
                dpi = 100
                figsize_x = 900 / dpi
                figsize_y = 500 / dpi
                for patron in patrones_encontrados:
                    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
                    plt.plot(patron.dataframe_segment['SMA'].tolist())
                    ax.plot(patron.dataframe_segment['SMA'].tolist())
                    # Title to plot
                    ax.set_title(f'{patron.company_name} - {patron.pattern_type}')
                    # Save the figure to a BytesIO object
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    buf.seek(0)
                    # save the image with the plot
                    img = Image.open(buf)
                    # generar numero aleatorio para el nombre de la imagen
                    img.save(f'{output_dir}{calcular_tiempo(FECHA)}_{random.randint(1, 1000)}.png') # numero de ventana, tipo de patron, distancia
                    buf.close()
                    plt.close(fig)
