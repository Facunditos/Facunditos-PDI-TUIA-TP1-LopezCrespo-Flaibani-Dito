import cv2
import numpy as np
import matplotlib.pyplot as plt

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

image = cv2.imread('examen_3.png')
np.unique(image)
len(np.unique(image))
imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
imshow(gray)

#--------------------------------
# Area Preguntas
#--------------------------------

# th=90
# image_th = image.copy()
# image_th[image_th > th] = 255
# imshow(image_th)

# image_th[image_th <= th] = 0
# imshow(image_th)

edges = cv2.Canny(gray, 100, 170, apertureSize=3)
# imshow(edges)

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
imshow(image_lines)
imshow(sheet)

header = image[0:lines_h_sorted[0][0][1], 0:image.shape[0]]
imshow(header)

questions = []
# Extraer las preguntas y agregarlas a la lista (columna izquierda)
for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
    question = image[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2:lines_v_sorted[2][0][0]-4]
    questions.append(question)

# Extraer las preguntas y agregarlas a la lista (columna derecha)
for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
    question = image[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[5][0][0]+2:lines_v_sorted[6][0][0]-4]
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
# imshow(questions[0])

# Hasta acá OK
#--------------------------------
# Area Respuestas
#--------------------------------
answers = []

for idx, question in enumerate(questions):
    question_gray = cv2.cvtColor(question, cv2.COLOR_BGR2GRAY)
    question_edges = cv2.Canny(question_gray, 100, 170, apertureSize=3)
    image_lines = question.copy()
    sheet = np.zeros(question.shape, dtype=np.uint8)
    lines = cv2.HoughLines(question_edges, rho=1, theta=np.pi/180, threshold=100)#22lineas los bordes
    # lines_v = []
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
        # if y1==1000:
        #     lines_v.append(((x1,y1),(x2,y2)))
        if x1==-1000:
            lines_h.append(((x1,y1),(x2,y2)))
        cv2.line(image_lines,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.line(sheet,(x1,y1),(x2,y2),(0,255,0),2)

    # Ordenar las listas porque Hough no genera las líneas en orden
    # lines_v_sorted = sorted(lines_v, key=lambda line: line[0][0])
    lines_h_sorted = sorted(lines_h, key=lambda line: line[0][1])

    # imshow(image_lines)
    # imshow(sheet)

    # answer = question[lines_h_sorted[0][0][1]-10:lines_h_sorted[0][0][1]+1, image_lines.shape[1]-150:image_lines.shape[1]]
    answer = question[lines_h_sorted[-1][0][1]-14:lines_h_sorted[-1][0][1]-1, 0:image_lines.shape[1]]
    answers.append(answer)
    

imshow(answers[1])

# Mostrar todas las preguntas en subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for idx, answer in enumerate(answers):
    # cv2.imwrite("Questions/" + 'question' +  str(idx + 1) + ".png", question, [cv2.IMWRITE_JPEG_QUALITY, 90])

    axes[idx].imshow(answer, cmap='gray')
    axes[idx].set_title(f'Pregunta {idx + 1}')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

imshow(answer[0])








#--------------------------------
# Componentes conectadas
#--------------------------------
# Forma 1
def letter_highlight(image):
    img = image.copy()
    imshow(img)
    mask = (img[:, :, 0] == img[:, :, 1]) & \
        (img[:, :, 1] == img[:, :, 2]) & \
        (img[:, :, 0] != 0)  # Excluir (0, 0, 0)
    # Cambiar los píxeles que cumplen con R=G=B a blanco (255, 255, 255)
    img[mask] = [255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th=250
    gray[gray > th] = 255
    gray[gray <= th] = 0
    return gray

j=0
imshow(questions[j])
image_th = letter_highlight(questions[j])
imshow(image_th)
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity, cv2.CV_32S)
# imshow(img=labels)

# Crear una copia de la imagen original (en color o escala de grises) para dibujar los rectángulos
output_image = cv2.cvtColor(image_th, cv2.COLOR_GRAY2BGR)

# Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
for i, st in enumerate(stats):
    x, y, w, h, area = st  # Obtener las estadísticas del componente actual
    if 0 < area < 600:
        margin = 5  # El valor de margen que quieres añadir
        x_new = max(x - margin, 0)  # Asegúrate de no ir más allá de los bordes de la imagen
        y_new = max(y - margin, 0)
        w_new = min(w + 2*margin, output_image.shape[1] - x_new)
        h_new = min(h + 2*margin, output_image.shape[0] - y_new)
        cv2.rectangle(output_image, (x_new, y_new), (x_new + w_new, y_new + h_new), color=(0, 255, 0), thickness=1)
        # cv2.rectangle(output_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # cv2.rectangle(output_image, (x-5, y-3), (x + w+3, y + h+2), color=(0, 255, 0), thickness=1)

# Mostrar la imagen con los rectángulos dibujados
imshow(img=output_image, color_img=True)


# Forma 2
j = 4
imshow(questions[j])
# Convertir la imagen a escala de grises
image_th = cv2.cvtColor(questions[j], cv2.COLOR_BGR2GRAY)
imshow(image_th)
# Realizar la operación de componentes conectadas
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity, cv2.CV_32S)
# Crear una copia de la imagen original (en color o escala de grises) para dibujar los rectángulos
output_image = cv2.cvtColor(image_th, cv2.COLOR_GRAY2BGR)

# Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
for i, st in enumerate(stats):
    x, y, w, h, area = st  # Obtener las estadísticas del componente actual
    if 0 < area < 1000:
        margin = 5  # El valor de margen que quieres añadir
        x_new = max(x - margin, 0)  # Asegúrate de no ir más allá de los bordes de la imagen
        y_new = max(y - margin, 0)
        w_new = min(w + 2*margin, output_image.shape[1] - x_new)
        h_new = min(h + 2*margin, output_image.shape[0] - y_new)
        # cv2.rectangle(output_image, (x_new, y_new), (x_new + w_new, y_new + h_new), color=(0, 255, 0), thickness=1)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # cv2.rectangle(output_image, (x-5, y-3), (x + w+3, y + h+2), color=(0, 255, 0), thickness=1)

# Mostrar la imagen con los rectángulos dibujados
imshow(img=output_image, color_img=True)


#Forma 3
j = 4
imshow(questions[j])
# Convertir la imagen a escala de grises
image_gray = cv2.cvtColor(questions[j], cv2.COLOR_BGR2GRAY)
_, image_th = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
imshow(image_th)
# Realizar la operación de componentes conectadas
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity, cv2.CV_32S)
# Crear una copia de la imagen original (en color o escala de grises) para dibujar los rectángulos
output_image = cv2.cvtColor(image_th, cv2.COLOR_GRAY2BGR)

# Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
for i, st in enumerate(stats):
    x, y, w, h, area = st  # Obtener las estadísticas del componente actual
    if 0 < area < 1000:
        margin = 5  # El valor de margen que quieres añadir
        x_new = max(x - margin, 0)  # Asegúrate de no ir más allá de los bordes de la imagen
        y_new = max(y - margin, 0)
        w_new = min(w + 2*margin, output_image.shape[1] - x_new)
        h_new = min(h + 2*margin, output_image.shape[0] - y_new)
        # cv2.rectangle(output_image, (x_new, y_new), (x_new + w_new, y_new + h_new), color=(0, 255, 0), thickness=1)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # cv2.rectangle(output_image, (x-5, y-3), (x + w+3, y + h+2), color=(0, 255, 0), thickness=1)

# Mostrar la imagen con los rectángulos dibujados
imshow(img=output_image, color_img=True)

# Forma 4
# def letter_highlight(image):
#     img = image.copy()
#     imshow(img)
#     mask = (img[:, :, 0] == img[:, :, 1]) & \
#         (img[:, :, 1] == img[:, :, 2]) & \
#         (img[:, :, 0] != 0)  # Excluir (0, 0, 0)
#     # Cambiar los píxeles que cumplen con R=G=B a blanco (255, 255, 255)
#     img[mask] = [255, 255, 255]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     th=250
#     gray[gray > th] = 255
#     gray[gray <= th] = 0
#     return gray

j=1
imshow(questions[j])
# image_th = letter_highlight(questions[j])
image_gray = letter_highlight(questions[j])
_, image_th = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
imshow(image_th)
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity, cv2.CV_32S)
# imshow(img=labels)

# Crear una copia de la imagen original (en color o escala de grises) para dibujar los rectángulos
output_image = cv2.cvtColor(image_th, cv2.COLOR_GRAY2BGR)

# Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
for i, st in enumerate(stats):
    x, y, w, h, area = st  # Obtener las estadísticas del componente actual
    if 0 < area < 600:
        margin = 5  # El valor de margen que quieres añadir
        x_new = max(x - margin, 0)  # Asegúrate de no ir más allá de los bordes de la imagen
        y_new = max(y - margin, 0)
        w_new = min(w + 2*margin, output_image.shape[1] - x_new)
        h_new = min(h + 2*margin, output_image.shape[0] - y_new)
        # cv2.rectangle(output_image, (x_new, y_new), (x_new + w_new, y_new + h_new), color=(0, 255, 0), thickness=1)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # cv2.rectangle(output_image, (x-5, y-3), (x + w+3, y + h+2), color=(0, 255, 0), thickness=1)

# Mostrar la imagen con los rectángulos dibujados
imshow(img=output_image, color_img=True)




