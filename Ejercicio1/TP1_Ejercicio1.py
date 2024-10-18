import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False, vmin=0, vmax=255):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray', vmin =vmin, vmax = vmax)
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def local_histogram_equalization(image, window_size):
    # Agregar borde a la imagen para manejar los bordes al mover la ventana
    top, bottom, left, right = window_size[0] // 2, window_size[0] // 2, window_size[1] // 2, window_size[1] // 2
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # Crear una imagen para almacenar el resultado
    output_image = np.zeros_like(image)

    # Obtener las dimensiones de la imagen
    rows, cols = image.shape

    # Recorrer cada píxel de la imagen
    for i in range(rows):
        for j in range(cols):
            # Extraer la ventana centrada en el píxel (i, j)
            window = padded_image[i:i+window_size[0], j:j+window_size[1]]

            # Calcular el histograma de la ventana
            hist, bins = np.histogram(window.flatten(), 256, [0, 256])

            # Calcular la función de distribución acumulada (CDF)
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf_normalized = cdf_normalized.astype('uint8')

            # Mapear el valor del píxel actual usando la CDF de la ventana
            output_image[i, j] = cdf_normalized[image[i, j]]

    return output_image

image = cv2.imread('Ejercicio1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Pruebas
np.unique(image)
len(np.unique(image))

prueba = image.copy()
prueba[prueba > 0] = 255
imshow(prueba)

prueba = image.copy()
prueba[prueba < 228] = 0
imshow(prueba)

prueba = image.copy()
prueba[prueba != 10] = 255
prueba[prueba == 10] = 0
imshow(prueba)

# Aplicar ecualización local del histograma con una ventana de tamaño 15x15
# window_size = (3, 3)
window_size = (25, 25)
equalized_image = local_histogram_equalization(image, window_size)

# Guardar el resultado
cv2.imwrite('Ejercicio1/Imagen_ecualizada_localmente.tif', equalized_image)

# Imagen original y la ecualizada
imshow(image)
title = '(' + str(window_size[0]) + ',' + str(window_size[1]) + ')'
imshow(equalized_image, title = title)


# Descripción del código:
# Bordes replicados: Para manejar el análisis en los bordes de la imagen, se añade un borde alrededor usando cv2.copyMakeBorder con la opción cv2.BORDER_REPLICATE, que replica el valor de los píxeles en los bordes.
# Ventana móvil: Se recorre cada píxel de la imagen y se extrae una ventana centrada en dicho píxel.
# Histograma y ecualización: Para cada ventana, se calcula el histograma y la CDF (Función de Distribución Acumulada), y se utiliza para mapear el valor del píxel centrado.
# Guardado del resultado: La imagen ecualizada localmente se guarda como un archivo TIFF.
# Análisis del tamaño de ventana:
# Tamaños de ventana más pequeños (por ejemplo, 3x3) pueden resaltar detalles muy locales, pero también pueden generar ruido.
# Tamaños de ventana más grandes (por ejemplo, 31x31) suavizan el resultado, pero pueden perder algunos detalles finos.
# Próximo paso:
# Puedes ajustar el tamaño de la ventana en el parámetro window_size para observar cómo cambia la calidad de los detalles resaltados
