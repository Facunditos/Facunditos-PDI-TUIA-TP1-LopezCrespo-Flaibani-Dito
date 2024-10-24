import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


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


img = cv2.imread('.Ejercicio1\Imagen_ecualizada_localmente_3x3.tif',cv2.IMREAD_GRAYSCALE)   
imshow(img)
img = cv2.imread('.Ejercicio1\Imagen_ecualizada_localmente_25x25.tif',cv2.IMREAD_GRAYSCALE)   
imshow(img)
window_size = 25,25
top, bottom, left, right = window_size[0] // 2, window_size[0] // 2, window_size[1] // 2, window_size[1] // 2
padded_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
imshow(padded_image)
img.shape
size = 9
size_2 = size*size
w1 = -np.ones((size,size))/size_2
w1[1,1]=size_2/size_2
img1 = cv2.filter2D(img,-1,w1)
imshow(img1)


img.shape
q_pix = img.size
unique, frequency = np.unique(img,return_counts=True)

unique
p_r = frequency / q_pix

hist, bins = np.histogram(img.flatten(),256,[0,256])
histn = hist.astype(np.double) / img.size
histn

def ecualizar_localmente(histograma:np.array):
    return

img_heq = cv2.equalizeHist(img)  
hist, bins = np.histogram(img_heq.flatten(),256,[0,256])
hist
img_1_sq = img[6:66,6:66]
img_1_sq_heq = cv2.equalizeHist(img_1_sq)  

imshow(img_1_sq)

ax1=plt.subplot(221)
plt.imshow(img_1_sq,cmap='gray',vmin=0,vmax=255)
plt.title('Imagen Original')

plt.subplot(222)
plt.hist(img_1_sq.flatten(), 256, [0, 256])
plt.title('Histograma')

plt.subplot(223,sharex=ax1,sharey=ax1)
plt.imshow(img_1_sq_heq,cmap='gray',vmin=0,vmax=255)
plt.title('Imagen Ecualizada')

plt.subplot(224)
plt.hist(img_1_sq_heq.flatten(), 256, [0, 256])
plt.title('Histograma')
plt.show()



