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
imshow(image)

# Pruebas
np.unique(image)
len(np.unique(image))

# Los pixeles que no son negros en la imagen, se pasan a blanco
prueba = image.copy()
prueba[prueba > 0] = 255
# imshow(prueba)

# Los pixeles que con valor menor a 228 en la imagen, se pasan a negro
prueba = image.copy()
prueba[prueba < 228] = 0
# imshow(prueba)

# prueba = image.copy()
# prueba[prueba != 10] = 255
# prueba[prueba == 10] = 0
# imshow(prueba)

# Aplicar ecualización local del histograma con una ventana de tamaño 19x19
window_size = (25, 25)
equalized_image = local_histogram_equalization(image, window_size)
cv2.imwrite('Ejercicio1/Imagen_ecualizada_localmente.tif', equalized_image)

plt.figure()
ax = plt.subplot(121)
imshow(image,new_fig=False, title="Imagen Original", colorbar=False)
plt.subplot(122, sharex=ax, sharey=ax), imshow(equalized_image, new_fig=False, title=f'Equalizada ({window_size[0]},{window_size[1]})', colorbar=False)
plt.show(block=False)

ax1=plt.subplot(221)
plt.imshow(image,cmap='gray',vmin=0,vmax=255)
plt.subplot(222)
plt.hist(image.flatten(), 256, [0, 256])
plt.subplot(223,sharex=ax1,sharey=ax1)
plt.imshow(equalized_image,cmap='gray',vmin=0,vmax=255)
plt.subplot(224)
plt.hist(equalized_image.flatten(), 256, [0, 256])
plt.show()

# Primer gráfico: Imagen Original
ax1 = plt.subplot(221)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Imagen Original')
# Segundo gráfico: Histograma de la imagen original
ax2 = plt.subplot(222)
plt.hist(image.flatten(), 256, [0, 256])
ax2.set_title('Histograma Imagen Original')
# Tercer gráfico: Imagen Equalizada
ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
ax3.set_title(f'Imagen Equalizada ({window_size[0]},{window_size[1]})')
# Cuarto gráfico: Histograma de la imagen equalizada
ax4 = plt.subplot(224)
plt.hist(equalized_image.flatten(), 256, [0, 256])
ax4.set_title('Histograma Imagen Equalizada')
plt.show()


# Aplicar ecualización local con distintos tamaños de ventana
window_sizes = [(3, 3), (5, 5), (9, 9), (13, 13), (19, 19), (25, 25)]

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.ravel()  # Para aplanar la matriz de subplots y poder iterar fácilmente

for idx, window_size in enumerate(window_sizes):
    equalized_image = local_histogram_equalization(image, window_size)

    axs[idx].imshow(equalized_image, cmap='gray')
    axs[idx].set_title(f'Window Size: {window_size}')
    axs[idx].axis('off')

    cv2.imwrite(f'Ejercicio1/Imagen_ecualizada_localmente_{window_size[0]}x{window_size[1]}.tif', equalized_image)

plt.tight_layout()
plt.show()


# Comparación con la ecualización global con cv2.equalizeHist()
ax1 = plt.subplot(221)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Imagen Original')
# Segundo gráfico: Histograma de la imagen original
ax2 = plt.subplot(222)
plt.hist(image.flatten(), 256, [0, 256])
ax2.set_title('Histograma Imagen Original')
# Tercer gráfico: Imagen Equalizada
ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
plt.imshow(img_heq, cmap='gray', vmin=0, vmax=255)
ax3.set_title(f'Imagen Equalizada - Global ({window_size[0]},{window_size[1]})')
# Cuarto gráfico: Histograma de la imagen equalizada
ax4 = plt.subplot(224)
plt.hist(img_heq.flatten(), 256, [0, 256])
ax4.set_title('Histograma Imagen Equalizada - Global')
plt.show()
