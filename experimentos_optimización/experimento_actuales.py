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

from pattern_utils import calculateSimpleMovingAverage, loadPatterns, findCommonPattern, acceptanceDistanceForPattern
from normalize_utils import normalizeVector
from tendencyCalculator import findPatternTendency
from experimento_15_v2 import Pattern
from pattern_search import findCurrentPatterns
# 1º) Programar una alternativa a findCurrentPatterns
# 2º) Comparar dicha alternativa con findCurrent original
TAMANIO_MINIMO_DATAFRAME = 15
MINIMUM_VALUE_TO_APPLY_SMA = 85
SMA_VALUE = 3

def searchPattern(company_dataframe, normalized_vector, patterns_dictionary, company_name):
  new_pattern_type, distance = findCommonPattern(normalized_vector, patterns_dictionary)
  new_pattern = None
  if distance < acceptanceDistanceForPattern(new_pattern_type):
    pattern_tendency = findPatternTendency(company_dataframe, company_dataframe, new_pattern_type)
    if pattern_tendency is not None:
       new_pattern = Pattern(new_pattern_type, company_dataframe, company_name, str(company_dataframe.iloc[0].name), str(company_dataframe.iloc[-1].name), None, distance, len(normalized_vector) ,pattern_tendency[2])
  return new_pattern

def acceptanceDistanceForPattern_findCommon(pattern_type):
  if pattern_type == 'double_top' or pattern_type == 'double_bottom':
    return 15
  elif pattern_type == 'inv_head_and_shoulders' or pattern_type == 'descesding_triangle':
    return 45
  elif pattern_type == 'head_and_shoulders':
    return 40
  elif pattern_type == 'ascending_triangle':
    return 35

def MIfindCurrentPatterns_configurable(company_dataframe, patterns_dictionary, company_name, attempt=3, failure_increment=0.3, 
                                                                                            acceptance_increment=0.1):
  # Mientras que la segunda ventana mejore la primera y se acepte la segunda ventana
  if (company_dataframe.shape[0] > MINIMUM_VALUE_TO_APPLY_SMA):
    company_dataframe = calculateSimpleMovingAverage(company_dataframe, SMA_VALUE)
    company_dataframe = company_dataframe.iloc[SMA_VALUE-1:]
  else:
    company_dataframe = calculateSimpleMovingAverage(company_dataframe, 1) # No le aplica la media movil  
  current_best_pattern = None
  while True:
    normalized_vector = normalizeVector(company_dataframe['SMA'].tolist())
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

def MIfindCurrentPatterns_automatico(company_name, patterns_dictionary):
  # Explorar varios tamaños de ventana que sean los que más patrones encuentran, para ello analizar datos experimentos
  # Explorar: 40, 140, 220
  windows_to_explore = [40, 140, 220]
  found_patterns = []
  for window_size in windows_to_explore:
    company_dataframe = get_company_data.getCompanyDataWithYahoo(company_name, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
    pattern_found = MIfindCurrentPatterns_configurable(company_dataframe, patterns_dictionary, company_name)
    if pattern_found is not None:
      found_patterns.append(pattern_found)
  return found_patterns


#patrones_a_estudiar = ['double_top', 'double_bottom', 'head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle', 'descending_triangle']
patrones_a_estudiar = ['head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle', 'descending_triangle']

window_sizes = [10, 20, 40, 80, 100, 120, 140, 160, 180, 200, 220, 230, 240, 250, 260, 280, 300, 320, 340, 360, 380, 400, 450, 500]
#window_sizes = [10, 20, 40, 80, 100, 120]

'''
companies = [
    "B", "BA", "BABA", "BAC", "BACA", "BACK", "BAER", "BAFN", "BAH", "BAK",
    "BALL", "BALY", "BAM", "BANC", "BAND", "BANF", "BANL", "BANR", "BANX",
    "BAOS", "BAP", "BARK", "BASE", "BATL", "BATRA", "BATRK", "BAX", "BAYA",
    "BB", "BBAI", "BBAR", "BBCP", "BBD", "BBDC", "BBDO", "BBGI", "BBIO",
    "BBLG", "BBSI", "BBU", "BBUC", "BBVA", "BBW", "BBWI", "BBY", "BC",
    "BCAB", "WLKP", "WLY", "WLYB", "WM", "WMB", "WMG", "WMK", "WMPN",
    "WMS", "WMT", "WNC", "WNEB", "WNS", "WNW", "WOLF", "WOOF",
    "WOR", "WORX", "WOW", "WPC", "WPM", "WPP", "WPRT", "WRAP",
    "WRB", "WRBY", "WRK", "WRLD", "WRN", "WRNT", "WS", "WSBC", "WSBF",
    "WSC", "WSFS", "WSM", "WSO", "WSR", "WST", "WT", "WTBA",
    "WTFC", "WTI", "WTM", "WTMA", "WTO", "WTRG", "WTS", "WTTR", "WTW",
    "WU", "WULF", "WVE", "WVVI", "WW", "WWD", "WWR", "WWW", "WY", "WYNN",
    "WYY", "X", "XAIR", "XBIO", "XBIT", "XBP", "XCUR", "XEL", "XELA",
    "XELB", "XENE", "XERS", "XFIN", "XFOR", "XGN", "XHR", "XIN", "XLO",
    "XMTR", "XNCR", "XNET", "XOM", "XOMA", "XOS", "XP", "XPEL",
    "XPER", "XPEV", "XPL", "XPO", "XPOF", "XPON", "XPRO", "XRAY", "XRTX",
    "XRX", "XTKG", "XTLB", "XTNT", "XWEL", "XXII", "XYF", "XYL", "YALA",
    "YCBD", "YELP", "YETI", "YEXT", "YGF", "YGMZ", "YHGJ", "YI", "YIBO",
    "YJ", "YMAB", "YMM", "YORW", "YOSH", "YOTA", "YOU", "YPF", "YQ", "YRD",
    "YS", "YSG", "YTEN", "YTRA", "YUM", "YUMC", "YY", "Z", "ZAPP", "ZBH",
    "ZBRA", "ZCAR", "ZCMD", "ZD", "ZDGE", "ZENV", "ZEPP", "ZETA", "ZEUS",
    "ZFOX", "ZG", "ZGN", "ZH", "ZI", "ZIM", "ZIMV", "ZION", "ZIP", "ZJYL",
    "ZKH", "ZKIN", "ZLAB", "ZLS", "ZM", "ZNTL", "ZOM", "ZS", "ZTEK", "ZTO",
    "ZTS", "ZUMZ", "ZUO", "ZURA", "ZVIA", "ZVRA", "ZVSA", "ZWS", "ZYME",
    "ZYXI", "CCOI", "CCRD", "CCRN", "CCS", "CCSI", "CCTG", "CCTS", "CCU",
    "CDAQ", "CDE", "CDIO", "CDLR", "CDLX", "CDMO", "CDNA", "CDNS", "CDP",
    "CDRE", "CDRO", "CDT", "CDTX", "CDW", "CDXC", "CDXS", "CDZI", "CE",
    "CEAD", "CECO", "CEG", "CEI", "CEIX", "CELC", "CELH", "CELU", "CELZ",
    "CENN", "CENT", "CENTA", "CENX", "CEPU", "CERE", "CERO", "CERS", "CERT",
    "CET", "CETU", "CETX", "CEVA", "CF", "CFB", "CFBK", "CFFI", "CFFN",
    "CFFS", "CFG", "CFLT", "CFR", "CFSB", "CG", "CGA", "CGAU", "CGBD",
    "CGC", "CGEM", "CGEN", "CGNT", "CGNX", "CGON", "CGTX", "CHAA", "CHCI",
    "CHCO", "CHCT", "CHD", "CHDN", "CHE", "CHEF", "CHEK", "CHGG", "CHH",
    "CHK", "CHKP", "CHMG", "CHMI", "CHNR", "CHPT", "CHR", "CHRD", "CHRO",
    "CHRS", "CHRW", "CHSN", "CHT", "CHTR", "CHUY", "CHWY", "CHX", "CI"
]
'''
companies = [
    "B", "BA", "BABA", "BAC", "BACA", "BACK", "BAER", "BAFN", "BAH", "BAK",
    "BALL", "BALY", "BAM", "BANC", "BAND", "BANF", "BANL", "BANR", "BANX",
    "BAOS", "BAP", "BARK", "BASE", "BATL", "BATRA", "BATRK", "BAX", "BAYA",
    "BB", "BBAI", "BBAR", "BBCP", "BBD", "BBDC", "BBDO", "BBGI", "BBIO",
    "BBLG", "BBSI", "BBUC", "BBVA", "BBW", "BBWI", "BBY", "BC",
    "BCAB", "WLKP", "WLY", "WLYB", "WM", "WMB", "WMG", "WMK", "WMPN", "WMS", "WMT", "WNC", "WNEB", "WNS", "WNW", "WOLF", "WOOF",
    "WOR", "WORX", "WOW", "WPC", "WPM", "WPP", "WPRT", "WRAP",
    "WRB", "WRBY", "WRK", "WRLD", "WRN", "WRNT", "WS", "WSBC", "WSBF",
    "WSC", "WSFS", "WSM", "WSO", "WSR", "WST", "WT", "WTBA",
    "WTFC", "WTI", "WTM", "WTMA", "WTO", "WTRG", "WTS", "WTTR", "WTW",
    "WU", "WULF", "WVE", "WVVI", "WW", "WWD", "WWR", "WWW", "WY", "WYNN",
    "WYY", "X", "XAIR", "XBIO", "XBIT", "XBP", "XCUR", "XEL", "XELA",
    "XELB", "XENE", "XERS", "XFIN", "XFOR", "XGN", "XHR", "XIN", "XLO",
    "XMTR", "XNCR", "XNET", "XOM", "XOMA", "XOS", "XP", "XPEL",
    "XPER", "XPEV", "XPL", "XPO", "XPOF", "XPON", "XPRO", "XRAY", "XRTX",
    "XRX", "XTKG", "XTLB", "XTNT", "XWEL", "XXII", "XYF", "XYL", "YALA"]


if __name__ == "__main__":
  input('Press enter to start the process')
  diccionario_patrones = loadPatterns(15, patrones_a_estudiar)
  df_patron = pd.DataFrame(columns=['Tipo_Ejecucion', 'Tamaño_Ventana', 'Tipo_Patron', 'Numero_encontrados', 'Distancia_media', 'Tiempo'])
  condicion = lambda patron, tipo: patron.pattern_type == tipo

  todos_mejorados = []
  tiempo_mejorado = 0
  for company in companies:
      print(f'Company: {company}')  
      
 
      start_time = time.time()
      pattern_improved = MIfindCurrentPatterns_automatico(company, diccionario_patrones)
      end_time = time.time()
      tiempo_mejorado += end_time - start_time
      if pattern_improved is not None:
        todos_mejorados = todos_mejorados + pattern_improved
  for patron in patrones_a_estudiar:
    if len([x for x in todos_mejorados if x.pattern_type == patron]) > 0:
      df_patron = df_patron._append({'Tipo_Ejecucion': 'Automático', 'Tamaño_Ventana': 0, 'Tipo_Patron': patron, 'Numero_encontrados': len([x for x in todos_mejorados if x.pattern_type == patron]), 'Distancia_media':  round(mean([ptr.distance for ptr in todos_mejorados if condicion(ptr, patron)]), 2), 'Tiempo': round(tiempo_mejorado / len(companies), 2)}, ignore_index=True)
    else:
      df_patron = df_patron._append({'Tipo_Ejecucion': 'Automático', 'Tamaño_Ventana': 0, 'Tipo_Patron': patron, 'Numero_encontrados': 0, 'Distancia_media': 0, 'Tiempo': round(tiempo_mejorado / len(companies), 2)}, ignore_index=True)
  df_patron.to_csv('./experimento_findCurrent/efc_automatico.csv', index=False)
  '''
  for window_size in window_sizes:
    print(f'Window size: {window_size}')
    tiempo_mejorado = 0
    tiempo_original = 0
    todos_originales = []
    todos_mejorados = []
    for company in companies:
      company_dataframe = get_company_data.getCompanyDataWithYahoo(company, (datetime.today() - timedelta(days=window_size)).strftime("%Y-%m-%d") ,datetime.today().strftime("%Y-%m-%d"))
      
      start_time = time.time()
      pattern_improved = MIfindCurrentPatterns_configurable(company_dataframe, diccionario_patrones, company)
      end_time = time.time()
      tiempo_mejorado += end_time - start_time
      if pattern_improved is not None:
        todos_mejorados.append(pattern_improved)

      #start_time = time.time()
      #pattern_original = findCurrentPatterns(company_dataframe, diccionario_patrones, company)
      #end_time = time.time()
      #tiempo_original += end_time - start_time
      #if len(pattern_original) > 0:
        #todos_originales.append(pattern_original[0])
    for patron in patrones_a_estudiar:
      if len([x for x in todos_originales if x.pattern_type == patron]) > 0:
        #print(f'Numero de elementos original {len([x for x in todos_originales if x.pattern_type == patron])}')
        df_patron = df_patron._append({'Tipo_Ejecucion': 'Original','Tamaño_Ventana': window_size, 'Tipo_Patron': patron, 'Numero_encontrados': len([x for x in todos_originales if x.pattern_type == patron]), 'Distancia_media':  round(mean([ptr.distance for ptr in todos_originales if condicion(ptr, patron)]), 2), 'Tiempo': round(tiempo_original / len(companies), 2)}, ignore_index=True)
      else:
        df_patron = df_patron._append({'Tipo_Ejecucion': 'Original','Tamaño_Ventana': window_size, 'Tipo_Patron': patron, 'Numero_encontrados': 0, 'Distancia_media': 0, 'Tiempo': round(tiempo_original / len(companies), 2)}, ignore_index=True)
      if len([x for x in todos_mejorados if x.pattern_type == patron]) > 0:
        #print(f'Numero de elementos mejorados {len([x for x in todos_mejorados if x.pattern_type == patron])}')
        df_patron = df_patron._append({'Tipo_Ejecucion': 'Mejorado','Tamaño_Ventana': window_size, 'Tipo_Patron': patron, 'Numero_encontrados': len([x for x in todos_mejorados if x.pattern_type == patron]), 'Distancia_media':  round(mean([ptr.distance for ptr in todos_mejorados if condicion(ptr, patron)]), 2), 'Tiempo': round(tiempo_mejorado / len(companies), 2)}, ignore_index=True)
      else:
        df_patron = df_patron._append({'Tipo_Ejecucion': 'Mejorado','Tamaño_Ventana': window_size, 'Tipo_Patron': patron, 'Numero_encontrados': 0, 'Distancia_media': 0, 'Tiempo': round(tiempo_mejorado / len(companies), 2)}, ignore_index=True)
  df_patron.to_csv('./experimento_findCurrent/efc_automatico.csv', index=False)
    '''


# Tipo_Ejecucion, Tamaño_Ventana, Numero_encontrados, Distancia_aceptacion, 