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


def ecualizar_localmente(img:np.array,tamaño_ventana)->np.array:
    top, bottom, left, right = tamaño_ventana[0] // 2, tamaño_ventana[0] // 2, tamaño_ventana[1] // 2, tamaño_ventana[1] // 2
    img_ampliada = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    n_filas, n_columnas = img_ampliada.shape
    img_ecualizada_localmente = np.copy(img_ampliada)
    min_idx_fila_interes = bottom
    max_idx_fila_interes = n_filas-bottom
    min_idx_col_interes = left
    max_idx_col_interes = n_columnas-bottom
    idx_fila_pixel_relevante = top +1
    idx_col_pixel_relevante = left +1
    for idx_fila in range(n_filas):
        for idx_columna in range(n_columnas):
            if ( (idx_fila>min_idx_fila_interes) and (idx_fila<max_idx_fila_interes) and (idx_columna>min_idx_col_interes) and (idx_columna<max_idx_col_interes) ): 
                subimagen = img_ampliada[ (idx_fila-top):(idx_fila+bottom+1),(idx_columna-left):(idx_columna+right+1) ]
                subimagen_ecualizada = cv2.equalizeHist(subimagen) 
                pixel_central_ecualizado = subimagen_ecualizada[idx_fila_pixel_relevante,idx_col_pixel_relevante]
                img_ecualizada_localmente[idx_fila,idx_columna] = pixel_central_ecualizado
    img_ecualizada_localmente = img_ecualizada_localmente[top:(n_filas-bottom) , left:(n_columnas-right) ]
    return  img_ecualizada_localmente


img = cv2.imread('.\Ejercicio1\Imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE)   
tamaño_ventana_chico = (3,3)
img_equ_ventana_chica = ecualizar_localmente(img,tamaño_ventana_chico)

tamaño_ventana_grande = (25,25)
img_equ_ventana_grande = ecualizar_localmente(img,tamaño_ventana_grande)

# Comparación originial contra ecualizada 3 por 3
ax1=plt.subplot(221)
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.title('Imagen Original')

plt.subplot(222)
plt.hist(img.flatten(), 256, [0, 256])
plt.title('Histograma imagen original')

plt.subplot(223,sharex=ax1,sharey=ax1)
plt.imshow(img_equ_ventana_chica,cmap='gray',vmin=0,vmax=255)
plt.title('Imagen Ecualizada Localmente. Ventana: 3*3')

plt.subplot(224)
plt.hist(img_equ_ventana_chica.flatten(), 256, [0, 256])
plt.title('Histograma Imagen Ecualizada Localmente. Ventana: 3*3')
plt.show(block=False)

# Comparación ecualizada 3 por 3 contra ecualizada 25 por 25
ax1=plt.subplot(221)
plt.imshow(img_equ_ventana_chica,cmap='gray',vmin=0,vmax=255)
plt.title('Imagen Ecualizada Localmente. Ventana: 3*3')

plt.subplot(222)
plt.hist(img_equ_ventana_chica.flatten(), 256, [0, 256])
plt.title('Histograma Imagen Ecualizada Localmente. Ventana: 3*3')

plt.subplot(223,sharex=ax1,sharey=ax1)
plt.imshow(img_equ_ventana_grande,cmap='gray',vmin=0,vmax=255)
plt.title('Imagen Ecualizada Localmente. Ventana: 25*25')

plt.subplot(224)
plt.hist(img_equ_ventana_grande.flatten(), 256, [0, 256])
plt.title('Histograma Imagen Ecualizada Localmente. Ventana: 25*25')
plt.show(block=False)




