import pandas as pd

# Hacer un script que para cada tabla de patrones ordene para cada rango de fecha cuales son las mejores tamaños de ventana
# y divisiones. Mostrando cuantos patrones se encontraron en cada caso. Siendo la mejor combinación la que más patrones encuentre

patterns_types = ['double_top', 'double_bottom', 'head_and_shoulders', 'inv_head_and_shoulders', 'ascending_triangle',
                       'descending_triangle']

def sort_data(df):
    df_sorted = df.sort_values(by=['rango_fecha', 'numero_encontrados_total'], ascending=[True, False])
    return df_sorted

def order_range_dates_alphabetically(df):
    for index, row in df.iterrows():
        date_range = row['rango_fecha']
        if date_range == '9 días (9)':
            df.at[index, 'rango_fecha'] = 'aa 9 días'
        elif date_range == '17 días (17)':
            df.at[index, 'rango_fecha'] = 'ab 17 días'
        elif date_range == '1 meses y 4 días (34)':
            df.at[index, 'rango_fecha'] = 'ac 1 meses y 4 días'
        elif date_range == '2 meses (60)':
            df.at[index, 'rango_fecha'] = 'ad 2 meses'
        elif date_range == '5 meses y 8 días (158)':
            df.at[index, 'rango_fecha'] = 'ae 5 meses y 8 días'
        elif date_range == '7 meses y 11 días (221)':
            df.at[index, 'rango_fecha'] = 'af 7 meses y 11 días'
        elif date_range == '9 meses y 2 días (272)':
            df.at[index, 'rango_fecha'] = 'ag 9 meses y 2 días'
        elif date_range == '1 años (365)':
            df.at[index, 'rango_fecha'] = 'ah 1 años'
        elif date_range == '1 años y 5 meses (515)':
            df.at[index, 'rango_fecha'] = 'ai 1 años y 5 meses'
        elif date_range == '2 años (730)':
            df.at[index, 'rango_fecha'] = 'aj 2 años'
        elif date_range == '3 años (1095)':
            df.at[index, 'rango_fecha'] = 'ak 3 años'
        elif date_range == '6 años y 1 días (2191)':
            df.at[index, 'rango_fecha'] = 'al 6 años y 1 días'
        elif date_range == '8 años y 2 días (2922)':
            df.at[index, 'rango_fecha'] = 'am 8 años y 2 días'
        elif date_range == '10 años y 2 días (3652)':
            df.at[index, 'rango_fecha'] = 'an 10 años y 2 días'
        elif date_range == '15 años y 3 días (5478)':
            df.at[index, 'rango_fecha'] = 'ao 15 años y 3 días'
    return df

def calculate_medium_foreach_date_range(df_10, df_15, df_20, df_25):
    df_medium = pd.DataFrame(columns=['num_patrones_cargados', 'rango_fecha', 'media_numero_encontrados_total', 'media_tiempo'])
    for df in [df_10, df_15, df_20, df_25]:
        df = df.drop(columns=['numero_ventanas_exploradas', 'total_aceptados_findCommon', 
                          'porcentaje_aceptacion_findCommon', 'porcentaje_aceptacion_tendency',
                          'indice'])
        df = sort_data(df)
        date_ranges = df['rango_fecha'].unique()
        for date_range in date_ranges:
            df_date_range = df.loc[df['rango_fecha'] == date_range]
            df_date_range = df_date_range.head(3)
            df_medium = df_medium._append({'num_patrones_cargados': df['num_patrones_cargar'].iloc[0], 'rango_fecha': date_range, 'media_numero_encontrados_total': round(df_date_range['numero_encontrados_total'].mean(), 2), 'media_tiempo': round(df_date_range['tiempo'].mean(), 2)}, ignore_index=True)
    return df_medium

def distances_and_lenght_for_windows_size_and_divisions(df_original, df_patterns):
    # Crear dataframe con columnas ['window_size_and_division','pattern_type', resto de columnas de por patrones, counter] con los valores a 0
    final_df = pd.DataFrame(columns=['tamano_ventana_y_division','tamano_ventana', 'divisiones', 'tipo_patron', 'numero_encontrados', 'distancia_promedio',
                                      'distancia_promedio_findCommon', 'num_aceptados_findCommon_patron', 'tamano_medio', 
                                      'porcentaje_aceptacion_tendency', 'contador_ocurrencias'])
    for indice in df_original['indice'].unique():
        window_size = df_original.loc[df_original['indice'] == indice]['tamano_ventana'].iloc[0]
        division = df_original.loc[df_original['indice'] == indice]['divisiones'].iloc[0]
        for pattern in patterns_types:
            ws_and_div = f'{window_size}_{division}'
            ws = window_size
            div = division
            if final_df[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern)].empty:
                final_df.loc[len(final_df)] = [ws_and_div, ws, div, pattern, 0, 0, 0, 0, 0, 0, 0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'contador_ocurrencias'] += 1 
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'numero_encontrados'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['numero_encontrados'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'distancia_promedio'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['distancia_promedio'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'distancia_promedio_findCommon'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['distancia_promedio_findCommon'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'num_aceptados_findCommon_patron'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['num_aceptados_findCommon_patron'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'tamano_medio'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['tamaño_medio'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'porcentaje_aceptacion_tendency'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['porcentaje_aceptacion_tendency'].iloc[0]

            else:
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'contador_ocurrencias'] += 1 
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'numero_encontrados'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['numero_encontrados'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'distancia_promedio'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['distancia_promedio'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'distancia_promedio_findCommon'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['distancia_promedio_findCommon'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'num_aceptados_findCommon_patron'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['num_aceptados_findCommon_patron'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'tamano_medio'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['tamaño_medio'].iloc[0]
                final_df.loc[(final_df['tamano_ventana_y_division'] == ws_and_div) & (final_df['tipo_patron'] == pattern), 'porcentaje_aceptacion_tendency'] += df_patterns.loc[(df_patterns['indice'] == indice) & (df_patterns['tipo_patron'] == pattern)]['porcentaje_aceptacion_tendency'].iloc[0]
    final_df['distancia_promedio'] = final_df['distancia_promedio'] / final_df['contador_ocurrencias']
    final_df['distancia_promedio_findCommon'] = final_df['distancia_promedio_findCommon'] / final_df['contador_ocurrencias']
    final_df['num_aceptados_findCommon_patron'] = final_df['num_aceptados_findCommon_patron'] / final_df['contador_ocurrencias']
    final_df['tamano_medio'] = final_df['tamano_medio'] / final_df['contador_ocurrencias']
    final_df['porcentaje_aceptacion_tendency'] = final_df['porcentaje_aceptacion_tendency'] / final_df['contador_ocurrencias']
    final_df = final_df.drop(columns=['contador_ocurrencias'])
    # redondea a dos decimales las columnas con datos
    final_df = final_df.round({'distancia_promedio': 2, 'distancia_promedio_findCommon': 2, 'num_aceptados_findCommon_patron': 2, 'tamano_medio': 2, 'porcentaje_aceptacion_tendency': 2})    
    # Para cada fila del nuevo dataframe dividir las columnas por counter  
    return final_df


    
if __name__ == "__main__":
    df_10 = pd.read_csv("tabla_general_10.csv")
    df_15 = pd.read_csv("tabla_general_15.csv")
    df_20 = pd.read_csv("tabla_general_20.csv")
    df_25 = pd.read_csv("tabla_general_25.csv")
    #calculate_medium_foreach_date_range(df_10, df_15, df_20, df_25).to_csv('./media_rango_fecha/datos_media.csv', index=False)
    df_15_v2 = pd.read_csv("tabla_general_15_v2.csv")
    df_15_p_v2 = pd.read_csv("tabla_por_patron_15_v2.csv")
    distances_and_lenght_for_windows_size_and_divisions(df_15_v2, df_15_p_v2).to_csv('./ventana_divisiones_por_patron_v2.csv', index=False)
    '''
    #calculate_medium_foreach_date_range(df_10, df_15, df_20, df_25).to_csv('./media_rango_fecha/datos_media.csv', index=False)
    #order_range_dates_alphabetically(sort_data(df_15)).to_csv('./media_rango_fecha/datos_15_patrones_general_ordenado.csv', index=False)
    df_15_by_pattern = pd.read_csv("tabla_por_patron_15.csv")
    distances_and_lenght_for_windows_size_and_divisions(df_15, df_15_by_pattern)
    '''



