import cv2
import numpy as np
import matplotlib.pyplot as plt



def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


img = cv2.imread('examen_5.png',cv2.IMREAD_GRAYSCALE)   
imshow(img)

#---------------Encabezado-----------------------#


th = img.max() * 0.9
img_th = img < th
img_rows = np.sum(img_th,1)
q_pix_unique = np.unique(img_rows)
q_pix_unique
th_row = 400
img_rows_th = img_rows > th_row
img_rows_th

idx_1_lin_h = np.argwhere(img_rows_th)[0,0]
img_enc = img_th[:(idx_1_lin_h+1),:]
imshow(img_enc,title='imagen binaria del encabezado')

#---------------Campos de Encabezados-----------------------#
#Se parte de una imagen booleana donde False es el fondo y True los objetos de interés

img_enc_rows = np.sum(img_enc,1)
np.unique(img_enc_rows)
max_q_pix = np.max(img_enc_rows)
# Identificamos en cuál índice horizontal se encuentra las líneas de los campos
idx_lin_campos_enc_h = np.argwhere(img_enc_rows==max_q_pix)[0,0]
lin_campos_enc_h = img_enc[idx_lin_campos_enc_h,:]

# Ahora pasamos a identificar los respectivos índices verticales 
y = np.diff(lin_campos_enc_h)
idxs_lin_campos_enc_v = np.argwhere(y)
idxs_lin_campos_enc_v
ii = np.arange(0,len(idxs_lin_campos_enc_v),2)    # 0 2 4 ... X --> X es el último nro par antes de len(renglones_indxs)
idxs_lin_campos_enc_v[ii]+=1
idxs_lin_campos_enc_v
idxs_lin_campos_enc_v = np.reshape(idxs_lin_campos_enc_v, (-1,2)) 
idxs_lin_campos_enc_v
idx_lin_campos_enc_h
# Defino la estructura de datos que va a contener toda la info relativa al campo de los encabezados
campos_enc = []
nombes_campos_enc = ['Name','Date','Class']
for i,nombre_campo in enumerate(nombes_campos_enc):
    # Creo la estructura de datos a utlizar para guardar la info relevante de cada campo
    datos_campo = {}
    datos_campo['nombre'] = nombre_campo
    idx_ini_lin_campos_enc_v = idxs_lin_campos_enc_v[i,0]
    idx_fin_lin_campos_enc_v = idxs_lin_campos_enc_v[i,1]
    # Cropping del campo del encabezado
    img_campo = img_enc[:idx_lin_campos_enc_h,idx_ini_lin_campos_enc_v:idx_fin_lin_campos_enc_v]
    # img_Campo el tipo de dato para poder utilizar la función de cv2 que no acepta datos booleanos
    img_campo = img_campo.astype('uint8')
    datos_campo['img'] = img_campo
    imshow(img_campo)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_campo, connectivity, cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
    #Excluimos al elemento encabezado para quedarnos únicamente con los caracteres
    estadisticas_caracteres = stats[1:,:]
    estadisticas_caracteres
    num_caracteres = len(estadisticas_caracteres)
    num_palabras = 1
    for i,stat_caracter in enumerate(estadisticas_caracteres):
        # El cálculo de la distancia lo empezamos a computar con el segundo caracter
        if (i==0):
            continue
        stat_anterior_caracter = estadisticas_caracteres[(i-1),:]
        x_final_anterior_caracter = stat_anterior_caracter[0] + stat_anterior_caracter[2]
        x_inicial_actual_caracter = stat_caracter[0]
        delta_h_entre_caracteres = x_inicial_actual_caracter - x_final_anterior_caracter
        # Si la distancia horizontal es mayor a tres píxeles el alumno escribió una nueva palabra 
        if (delta_h_entre_caracteres>4):
            num_palabras +=1
    datos_campo['num_caracteres'] = num_caracteres
    datos_campo['num_palabras'] = num_palabras
    campos_enc.append(datos_campo)

for campo_enc in campos_enc:
    nombre_campo = campo_enc['nombre']
    num_car = campo_enc['num_caracteres']
    num_palabras = campo_enc['num_palabras']
    img = campo_enc['img']
    if nombre_campo == 'Name':
        # Se corroborra si el alumno completó correctamente el nombre_campo
        respuesta = 'OK' if (num_car <= 25 and num_palabras >= 2) else 'MAL'
        campo_enc['respuesta'] = respuesta
    elif nombre_campo == 'Date':
        # Se corroborra si el alumno completó correctamente la fecha
        respuesta = 'OK' if (num_car == 8 and num_palabras == 1) else 'MAL'
        campo_enc['respuesta'] = respuesta
    elif nombre_campo == 'Class':
        # Se corroborra si el alumno completó correctamente la clase
        respuesta = 'OK' if (num_car == 1) else 'MAL'
        campo_enc['respuesta'] = respuesta
    print(f"{nombre_campo}:\t{respuesta}")  
    imshow(img, title=f"{nombre_campo}:{respuesta}")

print('fin de análisis del encabezado')