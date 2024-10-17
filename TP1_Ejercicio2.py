'''
De uno de los archivos examen.png se separa el encabezado: header y cada una de las 10 preguntas en una imagen distinta: questions[]
con el objetivo de detectar la zona de la respuesta
Falta obtener nombre y apellido, fecha? y clase
Falta armar una estructura por alumno con sus 10 respuestas y su nota
Falta comparar cada imagen de la respuesta y saber qué letra es??
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False, vmin = 0, vmax = 255):
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


def identify_letter(img):

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suavizar la imagen y aplicar binarización
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detectar contornos
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) == 2): return "B"
    elif (len(contours) == 0): return "C"
    elif (len(contours) == 1): #"A" o "D"

        return "B"
    else: return "X"
    # # Analizar contornos
    # for contour in contours:
    #     # Calcular el área y el perímetro
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)
        
    #     # Obtener la jerarquía (para contar contornos hijos, etc.)
    #     # En este caso, 'hierarchy' es una lista de la jerarquía de contornos
    #     # Puedes usarla para contar cuántos contornos hijos tiene cada uno.
        
    #     # Aproximar el contorno a un polígono
    #     epsilon = 0.02 * perimeter
    #     approx = cv2.approxPolyDP(contour, epsilon, True)
        
    #     # Mostrar los resultados
    #     print(f"Área: {area}, Perímetro: {perimeter}, Lados: {len(approx)}")
        
        # Comparar formas si tienes contornos de referencia (opcional)
        # hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        # Puedes comparar hu_moments con momentos de referencia

    # # Mostrar los contornos en la imagen
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("Contornos", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


#--------------------------------
image = cv2.imread('examen_3.png')
np.unique(image)
len(np.unique(image))
imshow(image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
imshow(image_gray)

image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
imshow(image_binary)


#--------------------------------
# Area Preguntas
#--------------------------------

edges = cv2.Canny(image_gray, 100, 170, apertureSize=3)
imshow(edges)

image_lines = image.copy()
sheet = np.zeros(image.shape, dtype=np.uint8)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=230)#22lineas los bordes
lines_v = []
lines_h = []
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    if y1==1000:
        lines_v.append(((x1,y1),(x2,y2)))
    if x1==-1000:
        lines_h.append(((x1,y1),(x2,y2)))
    cv2.line(image_lines,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.line(sheet,(x1,y1),(x2,y2),(0,255,0),2)

# Ordenar las listas porque Hough no genera las líneas en orden
lines_v_sorted = sorted(lines_v, key=lambda line: line[0][0])
lines_h_sorted = sorted(lines_h, key=lambda line: line[0][1])

imshow(image_lines)
imshow(sheet)

header = image_binary[0:lines_h_sorted[0][0][1], 0:image.shape[0]]
imshow(header)

questions = []
# Extraer las preguntas y agregarlas a la lista (columna izquierda)
for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
    question = image_binary[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2:lines_v_sorted[2][0][0]-4]
    questions.append(question)

# Extraer las preguntas y agregarlas a la lista (columna derecha)
for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
    question = image_binary[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[5][0][0]+2:lines_v_sorted[6][0][0]-4]
    questions.append(question)

# Mostrar todas las preguntas en subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for idx, question in enumerate(questions):
    cv2.imwrite("Questions/" + 'question' +  str(idx + 1) + ".png", question, [cv2.IMWRITE_JPEG_QUALITY, 90])

    axes[idx].imshow(question, cmap='gray')
    axes[idx].set_title(f'Pregunta {idx + 1}')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Hasta acá OK


#--------------------------------
# Area Respuestas
#--------------------------------

#--------------------------------
# Detección líneas debajo de la respuesta
# Componentes conectadas
#--------------------------------
fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # Crear una cuadrícula de 2 filas y 5 columnas

lines_answer = []
for j in range(10):

    # Realizar la operación de componentes conectadas
    connectivity = 8
    # Si la imagen no está invertida (fondo negro) no funciona: 255-questions[j]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-questions[j], connectivity, cv2.CV_32S)
    # Crear una copia de la imagen original (en color o escala de grises) para dibujar los rectángulos
    output_image = cv2.cvtColor(questions[j], cv2.COLOR_GRAY2BGR)

    # Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
    for i, st in enumerate(stats):
        x, y, w, h, area = st
        if w > 50 and h < 3:
            line_answer = questions[j][y-14:y+h-2, x:x+w]
            lines_answer.append(line_answer)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)

#     # Convertir la imagen de BGR (OpenCV) a RGB para mostrarla correctamente en Matplotlib
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Colocar la imagen en la subtrama correspondiente
    row = j // 5  # Calcular la fila (hay 2 filas)
    col = j % 5   # Calcular la columna (hay 5 columnas)
    axs[row, col].imshow(output_image_rgb)
    axs[row, col].axis('off')  # Ocultar los ejes para mejor visualización

plt.tight_layout()  # Ajustar el espaciado entre subtramas
plt.show()

# Mostrar todas las zonas de respuestas en subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for idx, line_answer in enumerate(lines_answer):
    cv2.imwrite("Lines/" + 'line_answer' +  str(idx + 1) + ".png", line_answer, [cv2.IMWRITE_JPEG_QUALITY, 90])

    axes[idx].imshow(line_answer, cmap='gray')
    axes[idx].set_title(f'Linea {idx + 1}')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

#--------------------------------
# Verificar que haya solo una letra
# Componentes conectados
#--------------------------------
k=1
imshow(lines_answer[k])

letter = ""
connectivity = 8
# Si la imagen no está invertida (fondo negro) no funciona: 255-questions[j]
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-lines_answer[k], connectivity, cv2.CV_32S)
if num_labels == 2:
#--------------------------------
# Segmentar las letras
# Obtención de contornos
#--------------------------------

# answers[0] C contorno sin hijos
# answers[1] B contorno con 2 hijos
# answers[2] A contorno con 1 hijo / relación area_hijo/area_padre < 0.65
# answers[3] D contorno con 1 hijo / relación area_hijo/area_padre > 0.65


    # Si la imagen no está invertida (fondo negro) no funciona: 255-lines_answer[k]
    # contornos, jerarquia = cv2.findContours(255-lines_answer[k], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos, jerarquia = cv2.findContours(255-lines_answer[k], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    siguiente, anterior, primer_hijo, padre = jerarquia[0][0]

    # Si primer_hijo es -1, no tiene hijos; si es distinto de -1, tiene un hijo.
    num_hijos = 0
    area_hijo=0
    area_padre=0
    aspect_ratio=0
    if primer_hijo != -1:
        print(f"Contorno 0 tiene un hijo con índice {primer_hijo}")

        # Ahora contamos cuántos hijos tiene contornos[0]
        # num_hijos = 0
        hijo = primer_hijo
        while hijo != -1:
            num_hijos += 1
            siguiente_hijo = jerarquia[0][hijo][0]  # Siguiente contorno en el mismo nivel
            hijo = siguiente_hijo

        if num_hijos==2: letter="B"
        elif num_hijos==1:
            # Diferenciar entre "A" y "D"
            x, y, w, h = cv2.boundingRect(contornos[primer_hijo])  # Obtener el rectángulo delimitador del contorno hijo
            aspect_ratio = w / h  # Relación de aspecto
            area_hijo = cv2.contourArea(contornos[primer_hijo])  # Área del contorno hijo
            area_padre = cv2.contourArea(contornos[0])  # Área del contorno padre

            if area_hijo / area_padre < 0.65: # and aspect_ratio < 1:
                letter = "A"  # El hueco es más pequeño y centrado
            else:
                letter = "D"  # El hueco es más grande y alargado

        # else: letter='X'
    else: letter="C"

print(letter)


# para segmentar las letras directamente de cada imagen de pregunta
# No detecta los límites exactos de la letra supuestamente por el procesamiento previo de la imagen para tener una imagen binaria
# hay 4 pruebas distintas de procesamiento previo de la imagen
#--------------------------------

# question= questions[0]
# imshow(question)
# question.shape

# question_zeros = question == 0
# question_zeros.

# Preprocesamiento
# answers[0] C 1 contorno
# answers[1] B 2 contornos 2 2 0
# answers[2] A 2 contornos
# answers[3] D 1 contorno 1 1 0
# Función para verificar si un contorno está cerrado
# def es_contorno_cerrado(contorno):
#     # El contorno está cerrado si el primer punto y el último son iguales
#     return (contorno[0][0] == contorno[-1][0]).all()

# k=3
# # gray = cv2.cvtColor(answers[k], cv2.COLOR_BGR2GRAY)
# # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# _, img_le = cv2.threshold(answers[k], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # img_le = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# # img_le = cv2.GaussianBlur(img_le, (5, 5), 0)
# imshow(answers[k])
# # imshow(gray)
# # imshow(blurred)
# imshow(img_le)
# # Obtención de contornos
# contornos, jerarquia = cv2.findContours(img_le, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# len(contornos)
# cc=0
# ca=0
# for contorno in contornos:
#     if es_contorno_cerrado(contorno):
#         cc+=1
#     else:
#         ca+=1
# print(len(contornos), ca, cc)

# if len(contornos)>=2:
#     letra='B'


# siguiente, anterior, primer_hijo, padre = jerarquia[0][0]

# # Si primer_hijo es -1, no tiene hijos; si es distinto de -1, tiene un hijo.
# if primer_hijo != -1:
#     print(f"Contorno 0 tiene un hijo con índice {primer_hijo}")
    
#     # Ahora contamos cuántos hijos tiene contornos[0]
#     num_hijos = 0
#     hijo = primer_hijo
#     while hijo != -1:
#         num_hijos += 1
#         siguiente_hijo = jerarquia[0][hijo][0]  # Accedemos al siguiente contorno en el mismo nivel
#         hijo = siguiente_hijo

#     print(f"Contorno 0 tiene {num_hijos} hijos.")
# else:
#     print("Contorno 0 no tiene hijos.")

# # Análisis de características
# caracteristicas = []
# for contorno in contornos:
#     area = cv2.contourArea(contorno)
#     x, y, w, h = cv2.boundingRect(contorno)
#     relacion_aspecto = float(w)/h
#     cantidad_contornos_hijos = len(cv2.findContours(contorno, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0])
#     print(area, relacion_aspecto, cantidad_contornos_hijos)
#     caracteristicas.append([area, relacion_aspecto, cantidad_contornos_hijos])


# Forma 2
# Imagen en gris
fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # Crear una cuadrícula de 2 filas y 5 columnas

answers = []
for j in range(10):
    # imshow(questions[j])
    # Convertir la imagen a escala de grises
    image_th1 = cv2.cvtColor(questions[j], cv2.COLOR_BGR2GRAY)
    # image_th = image_th1>100
    th=150
    image_th = image_th1.copy()
    image_th[image_th1 > th] = 255
    image_th[image_th1 <= th] = 0
    imshow(image_th)
    # Realizar la operación de componentes conectadas
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity, cv2.CV_32S)
    # Crear una copia de la imagen original (en color o escala de grises) para dibujar los rectángulos
    # output_image = cv2.cvtColor(image_th, cv2.COLOR_GRAY2BGR)

    # Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
    for i, st in enumerate(stats):
        x, y, w, h, area = st
        if 0 < area < 1000:
            margin = 5  # El valor de margen
            # x_new = max(x - margin, 0)
            # y_new = max(y - margin, 0)
            # w_new = min(w + 2*margin, output_image.shape[1] - x_new)
            # h_new = min(h + 2*margin, output_image.shape[0] - y_new)
            answer = image_th[y:y+h, x:x+w]
            # imshow(letter)
            answers.append(answer)
            # cv2.rectangle(output_image, (x_new, y_new), (x_new + w_new, y_new + h_new), color=(0, 255, 0), thickness=1)
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
            # cv2.rectangle(output_image, (x-5, y-3), (x + w+3, y + h+2), color=(0, 255, 0), thickness=1)

#     # Convertir la imagen de BGR (OpenCV) a RGB para mostrarla correctamente en Matplotlib
#     output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

#     # Colocar la imagen en la subtrama correspondiente
#     row = j // 5  # Calcular la fila (hay 2 filas)
#     col = j % 5   # Calcular la columna (hay 5 columnas)
#     axs[row, col].imshow(output_image_rgb)
#     axs[row, col].axis('off')  # Ocultar los ejes para mejor visualización

# plt.tight_layout()  # Ajustar el espaciado entre subtramas
# plt.show()
# Mostrar todas las preguntas en subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for idx, answer in enumerate(answers):
    cv2.imwrite("Answers/" + 'answer' +  str(idx + 1) + ".png", answer, [cv2.IMWRITE_JPEG_QUALITY, 90])

    axes[idx].imshow(answer, cmap='gray')
    axes[idx].set_title(f'Respuesta {idx + 1}')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


#-------------------
# Estructura
#-------------------

def create_questions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    imshow(gray)

    edges = cv2.Canny(gray, 100, 170, apertureSize=3)

    image_lines = image.copy()
    sheet = np.zeros(image.shape, dtype=np.uint8)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=230)#22lineas los bordes
    lines_v = []
    lines_h = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+1000*(-b))
        y1=int(y0+1000*(a))
        x2=int(x0-1000*(-b))
        y2=int(y0-1000*(a))
        if y1==1000:
            lines_v.append(((x1,y1),(x2,y2)))
        if x1==-1000:
            lines_h.append(((x1,y1),(x2,y2)))
        cv2.line(image_lines,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.line(sheet,(x1,y1),(x2,y2),(0,255,0),2)

    # Ordenar las listas porque Hough no genera las líneas en orden
    lines_v_sorted = sorted(lines_v, key=lambda line: line[0][0])
    lines_h_sorted = sorted(lines_h, key=lambda line: line[0][1])

    # imshow(image_lines, color_img=True)
    # imshow(sheet, color_img=True)
    # imshow(image_lines)
    # imshow(sheet)

    header = image[0:lines_h_sorted[0][0][1], 0:image.shape[0]]
    header_coords = (0,lines_h_sorted[0][0][1], 0,image.shape[0])
    imshow(header)

    questions = []
    questions_coords = []
    # Extraer las preguntas y agregarlas a la lista (columna izquierda)
    for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
        question = image[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2:lines_v_sorted[2][0][0]-4]
        questions.append(question)
        #(x, x+w, y, y+h)
        questions_coords.append(lines_h_sorted[i][0][1]+2,lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2,lines_v_sorted[2][0][0]-4))

    # Extraer las preguntas y agregarlas a la lista (columna derecha)
    for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
        question = image[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[5][0][0]+2:lines_v_sorted[6][0][0]-4]
        questions.append(question)
        #(x, x+w, y, y+h)
        questions_coords.append(lines_h_sorted[i][0][1]+2,lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2,lines_v_sorted[2][0][0]-4))


    for idx, question in enumerate(questions):
        cv2.imwrite("Questions/" + 'question' +  str(idx + 1) + ".png", question, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return header, header_coords, questions, questions_coords

def pupil():
    pass

def pupil_score():
    pass

dir_path = 'Tests/'

# Separa cada pregunta por única vez
questions = []
questions_coords = []
header, header_coords, questions, questions_coords = create_questions(dir_path + 'examen_1.png')


# Analiza cada examen

# imagenes = []

for file in os.listdir(dir_path):
    if file.endswith('.png'):  # Solo procesar archivos PNG
        # Lee la imagen con OpenCV
        image = cv2.imread(os.path.join(dir_path, file))
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            name, clas = pupil(header)
            for coord in questions_coords:
                scores = pupil_score()

#             imagenes.append(gray)
# print(f'Se han procesado {len(imagenes)} imágenes')


#---------------
# Notas
#---------------
# probar invertir la imagen al llamar a cv2.connectedComponentsWithStats
# la función cv2.connectedComponentsWithStats en OpenCV generalmente espera imágenes binarias 
# donde el fondo sea negro (valor 0) y los objetos o componentes sean blancos (valor 255).


# Cargar la imagen en escala de grises
image = cv2.imread('imagen.png', cv2.IMREAD_GRAYSCALE)

# Invertir la imagen (hacer el negativo)
inverted_image = cv2.bitwise_not(image)

# Mostrar la imagen invertida
cv2.imshow('Imagen Invertida', inverted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#---------------

# Cargar la imagen en escala de grises
image = cv2.imread('imagen.png', cv2.IMREAD_GRAYSCALE)

# Invertir la imagen (hacer el negativo) restando de 255
inverted_image = 255 - image

# Mostrar la imagen invertida
cv2.imshow('Imagen Invertida', inverted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#---------------
# Comparación de imágenes para saber qué letra seleccionó cada alumno
#---------------

# 1. Comparar imágenes por diferencias pixel a pixel
# Esta técnica simplemente calcula la diferencia absoluta entre los valores de los píxeles de las dos imágenes. 
# Si las imágenes son iguales, la diferencia será cero para todos los píxeles.
import cv2
import numpy as np

# Cargar las imágenes
img1 = cv2.imread('imagen1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imagen2.png', cv2.IMREAD_GRAYSCALE)

# Restar las imágenes
diferencia = cv2.absdiff(img1, img2)

# Mostrar la diferencia
cv2.imshow('Diferencia', diferencia)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calcular la cantidad de píxeles diferentes
if np.count_nonzero(diferencia) == 0:
    print("Las imágenes son iguales")
else:
    print("Las imágenes son diferentes")


# 2. Comparar usando histogramas
# Puedes comparar las imágenes basándote en sus histogramas, que representan la distribución de los píxeles. 
# OpenCV tiene la función cv2.compareHist que te permite comparar histogramas y determinar cuán similares son dos imágenes.
import cv2

# Cargar las imágenes
img1 = cv2.imread('imagen1.png')
img2 = cv2.imread('imagen2.png')

# Convertir las imágenes a escala de grises
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calcular los histogramas
hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

# Normalizar los histogramas
hist1 = cv2.normalize(hist1, hist1)
hist2 = cv2.normalize(hist2, hist2)

# Comparar los histogramas
comparacion = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
print(f"Similitud de las imágenes: {comparacion}")

# 3. Comparar usando características (ORB, SIFT, o SURF)
# Este método consiste en extraer características de las imágenes y luego compararlas. 
# Puedes utilizar detectores como ORB, SIFT o SURF (aunque SIFT y SURF requieren instalar OpenCV con contrib).
# Aquí tienes un ejemplo usando ORB (que es rápido y no requiere una versión especial de OpenCV):

import cv2

# Cargar las imágenes
img1 = cv2.imread('imagen1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imagen2.png', cv2.IMREAD_GRAYSCALE)

# Detectar y describir características con ORB
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Usar un matcher para comparar los descriptores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Ordenar las coincidencias por distancia
matches = sorted(matches, key=lambda x: x.distance)

# Mostrar las mejores coincidencias
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Coincidencias', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
