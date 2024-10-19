#-------------------
# Librerías
#-------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

#-------------------
# Funciones
#-------------------

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

'''
Detecta y separa las zonas de cada pregunta en la imagen
'''
def create_questions(image: np.ndarray) -> tuple:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(image_gray, 100, 170, apertureSize=3)

    # image_lines = image.copy()
    # sheet = np.zeros(image.shape, dtype=np.uint8)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=230)#22 lineas los bordes
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
        # cv2.line(image_lines,(x1,y1),(x2,y2),(0,255,0),2)
        # cv2.line(sheet,(x1,y1),(x2,y2),(0,255,0),2)

    # Ordenar las listas porque Hough no genera las líneas en orden
    lines_v_sorted = sorted(lines_v, key=lambda line: line[0][0])
    lines_h_sorted = sorted(lines_h, key=lambda line: line[0][1])

    # FACUNDO si no querés tener la lína abajo, sacale los 2 "+ 5". Te dejé la línea para que detectes las zonas de name, date y class
    header = image_binary[0:lines_h_sorted[0][0][1] + 5, 0:image.shape[0]]
    header_coords = (0,lines_h_sorted[0][0][1] + 5, 0,image.shape[0])
    # imshow(header)

    questions = []
    questions_coords = []
    # Extraer las preguntas y agregarlas a la lista (columna izquierda)
    for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
        question = image_binary[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2:lines_v_sorted[2][0][0]-4]
        questions.append(question)
        #(x, x+w, y, y+h)
        questions_coords.append((lines_h_sorted[i][0][1]+2,lines_h_sorted[i+1][0][1]-4, lines_v_sorted[1][0][0]+2,lines_v_sorted[2][0][0]-4))

    # Extraer las preguntas y agregarlas a la lista (columna derecha)
    for i in range(3, 13, 2):  # Va de la línea 3 a la 12 (pares) para obtener las filas
        question = image_binary[lines_h_sorted[i][0][1]+2:lines_h_sorted[i+1][0][1]-4, lines_v_sorted[5][0][0]+2:lines_v_sorted[6][0][0]-4]
        questions.append(question)
        #(x, x+w, y, y+h)
        questions_coords.append((lines_h_sorted[i][0][1]+2,lines_h_sorted[i+1][0][1]-4, lines_v_sorted[5][0][0]+2,lines_v_sorted[6][0][0]-4))

    # # Guarda las imágenes en Questions/
    # for idx, question in enumerate(questions):
    #     cv2.imwrite("Ejercicio2/Questions/" + 'question' +  str(idx + 1) + ".png", question, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return header, header_coords, questions, questions_coords

'''
Dentro de cada pregunta del examen, detecta y separa las zonas de cada de las respuestas
'''
def identify_lines(questions: list) -> tuple:
    lines_answer = []
    lines_answer_coords = []
    for question in questions:
        # question = image_binary[coords[0]:coords[1], coords[2]:coords[3]]

        # Realizar la operación de componentes conectadas
        connectivity = 8
        # Si la imagen no está invertida (fondo negro) no funciona: 255-questions[j]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-question, connectivity, cv2.CV_32S)

        # Iterar sobre las estadísticas y dibujar los rectángulos para los componentes con área en el rango deseado
        for i, st in enumerate(stats):
            x, y, w, h, area = st
            if w > 50 and h < 3:
                line_answer = question[y-14:y+h-2, x:x+w]
                lines_answer.append(line_answer)
                lines_answer_coords.append((y-14, y+h-2, x, x+w))

    # # Guarda las imágenes en Answers/
    # for idx, line_answer in enumerate(lines_answer):
    #     cv2.imwrite("Ejercicio2/Answers/" + 'answer' + str(idx + 1) + ".png", line_answer, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return lines_answer, lines_answer_coords

'''
Dentro de las zonas de respuesta, identifica si la respuesta tiene un formato válido de una letra
Devuelve todas las respuestas de un exámen
'''
def indetify_answers(lines_answer: list) -> list:
    answers = []
    for line_answer in lines_answer:
        letter = ""
        #--------------------------------
        # Verificar que haya solo una letra: num_labels == 2 (contorno de la imagen + letra)
        # Componentes conectados
        #--------------------------------
        connectivity = 8
        # Si la imagen no está invertida (fondo negro) no funciona: 255-questions[j]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-line_answer, connectivity, cv2.CV_32S)
        if num_labels == 2:
            #--------------------------------
            # Segmentar las letras
            # Obtención de contornos

            # answers[0] C contorno sin hijos
            # answers[1] B contorno con 2 hijos
            # answers[2] A contorno con 1 hijo / relación area_hijo/area_padre < 0.65
            # answers[3] D contorno con 1 hijo / relación area_hijo/area_padre > 0.65
            #--------------------------------

            # Si la imagen no está invertida (fondo negro) no funciona: 255-lines_answer[k]
            # contornos, hierarchy = cv2.findContours(255-lines_answer[k], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(255-line_answer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            next, previuos, first_child, father = hierarchy[0][0]

            # Si first_child es -1, no tiene hijos; si es distinto de -1, tiene un hijo.
            child_count = 0
            child_area=0
            father_area=0
            aspect_ratio=0
            if first_child != -1:
                # print(f"Contorno 0 tiene un hijo con índice {first_child}")
                # Ahora contamos cuántos hijos tiene contours[0]
                # child_count = 0
                child = first_child
                while child != -1:
                    child_count += 1
                    next_child = hierarchy[0][child][0]  # Siguiente contorno en el mismo nivel
                    child = next_child

                if child_count==2: letter="B"
                elif child_count==1:
                    # Diferenciar entre "A" y "D"
                    x, y, w, h = cv2.boundingRect(contours[first_child])  # Obtener el rectángulo delimitador del contorno hijo
                    aspect_ratio = w / h  # Relación de aspecto
                    child_area = cv2.contourArea(contours[first_child])  # Área del contorno hijo
                    father_area = cv2.contourArea(contours[0])  # Área del contorno padre

                    if child_area / father_area < 0.65: # and aspect_ratio < 1:
                        letter = "A"  # El hueco es más pequeño y centrado
                    else:
                        letter = "D"  # El hueco es más grande y alargado

            else: letter="C"

        answers.append(letter)

    return answers

'''
Muestra en pantalla el resultado detallado de cada exámen
'''
def results_to_screen(exams: list) -> None:
    for exam in exams:
        print(f"\nId: {exam['id']}")
        print(f"\nNombre: {exam['name']}")
        print(f"Fecha: {exam['date']}")
        print(f"Clase: {exam['class']}")
        print("Respuestas:")

        # Recorrer las respuestas y mostrar si son OK o MAL
        for idx, (key, value) in enumerate(exam["answers"].items(), 1):
            print(f"Pregunta {idx}: {value['state']}")

        # Mostrar si aprobó o no
        print(f"Resultado: {exam['passed']}")

'''
Guarda los exámenes corregidos en un archivo CSV
'''
def results_to_csv(path: str, exams: list) -> None:
    # output_csv = path + '/resultados_examenes.csv'
    output_csv = os.path.join(path, 'resultados_examenes.csv')

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Escribir encabezados
        writer.writerow(["Id", "Nombre", "Fecha", "Clase", "Pregunta", "Respuesta", "Estado", "Resultado"])

        for exam in exams:
            for idx, (key, value) in enumerate(exam["answers"].items(), 1):
                writer.writerow([exam["id"], exam["name"], exam["date"], exam["class"], idx, value["answer"], value["state"], exam["passed"]])


'''
Guarda los exámenes corregidos en un archivo TXT
'''
def results_to_txt(path: str, exams: list) -> None:
    # output_file = path + '/resultados_examenes.txt'
    output_file = os.path.join(path, 'resultados_examenes.txt')

    with open(output_file, 'w') as f:
        for exam in exams:
            f.write(f"\nId: {exam['id']}\n")
            f.write(f"\nNombre: {exam['name']}\n")
            f.write(f"Fecha: {exam['date']}\n")
            f.write(f"Clase: {exam['class']}\n")
            f.write("Respuestas:\n")

            # Recorrer las respuestas y mostrar si son OK o MAL
            for idx, (key, value) in enumerate(exam["answers"].items(), 1):
                f.write(f"Pregunta {idx}: {value['state']}\n")

            # Mostrar si aprobó o no
            f.write(f"Resultado: {exam['passed']}\n")

#-------------------
# Estructura
#-------------------
def main():
    print("*** Corrección de exámenes ***")
    dir_path = input("\nIngrese la carpeta que contiene los exámenes a corregir ('Ejercicio2/Tests/'): ")

        # Verificar si el directorio existe
    if not os.path.exists(dir_path):
        print(f"Error: El directorio '{dir_path}' no existe.")
        return  # Terminar el programa si el directorio no se encuentra

    # Variable para verificar si se encontraron archivos PNG
    found_png_files = False

    right_answers = ["C", "B", "A", "D", "B", "B", "A", "B", "D", "D"]
    answers = []
    exams = []

    # Analizar cada examen
    for file in os.listdir(dir_path):
        if file.endswith('.png'):  # Solo procesar archivos PNG
            found_png_files = True

            image = cv2.imread(os.path.join(dir_path, file))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            if image is not None:
                # Procesar encabezado y preguntas
                header, header_coords, questions, questions_coords = create_questions(image)

                # FACUNDO
                # ACA Trabajar el header (nombre, fecha, clase)
                    # Name: OK/MAL
                    # Date: OK/MAL
                    # Class: OK/MAL
                id = file
                name = "OK"
                date = "OK"
                class_n = "MAL"

                # Identificar las respuestas
                lines_answer, lines_answer_coords = identify_lines(questions)
                answer = indetify_answers(lines_answer)
                answers.append(answer)

                # Crear un nuevo diccionario para el examen actual
                exam = {
                    "id": id,
                    "name": name,
                    "date": date,
                    "class": class_n,
                    "answers": {},
                    "passed": ""
                }

                # Cargar las respuestas y verificar si son correctas
                for idx, item in enumerate(answer):
                    if item == right_answers[idx]:
                        exam["answers"]["answer_" + str(idx + 1)] = {"answer": item, "state": "OK"}
                    else:
                        exam["answers"]["answer_" + str(idx + 1)] = {"answer": item, "state": "MAL"}

                # Comparar las respuestas correctas con las dadas para determinar si aprobó
                score = sum(1 for ra, a in zip(right_answers, answer) if ra == a)
                if score >= 6:  # 6 es el umbral de aprobación
                    exam["passed"] = "APR"
                else:
                    exam["passed"] = "NO APR"

                exams.append(exam)

    # Verificar si no se encontraron archivos PNG
    if not found_png_files:
        print("No se encontraron archivos PNG en el directorio especificado.")

    results_to_screen(exams)
    results_to_csv(dir_path, exams)
    results_to_txt(dir_path, exams)

if __name__ == "__main__":
    main()

# # Analiza un solo exámen
# print("*** Corrección de exámenes ***")
# dir_path = input("\nIngrese el examen a corregir ('Ejercicio2/Tests/Examen_1.png'): ")
# directory, file = os.path.split(dir_path)
# file_name, _ = os.path.splitext(file)

# right_answers = ["C", "B", "A", "D", "B", "B", "A", "B", "D", "D"]
# answers = []
# exams = []

# image = cv2.imread(dir_path)
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# if image is not None:
#     # Procesar encabezado y preguntas
#     header, header_coords, questions, questions_coords = create_questions(image)

#     # FACUNDO
#     # ACA Trabajar el header (nombre, fecha, clase)
#         # Name: OK/MAL
#         # Date: OK/MAL
#         # Class: OK/MAL
#     id = file_name
#     name = "OK"
#     date = "OK"
#     class_n = "MAL"

#     # Identificar las respuestas
#     lines_answer, lines_answer_coords = identify_lines(questions)
#     answer = indetify_answers(lines_answer)
#     answers.append(answer)

#     # Crear un nuevo diccionario para el examen actual
#     exam = {
#         "id": id,
#         "name": name,
#         "date": date,
#         "class": class_n,
#         "answers": {},
#         "passed": ""
#     }

#     # Cargar las respuestas y verificar si son correctas
#     for idx, item in enumerate(answer):
#         if item == right_answers[idx]:
#             exam["answers"]["answer_" + str(idx + 1)] = {"answer": item, "state": "OK"}
#         else:
#             exam["answers"]["answer_" + str(idx + 1)] = {"answer": item, "state": "MAL"}

#     # Comparar las respuestas correctas con las dadas para determinar si aprobó
#     score = sum(1 for ra, a in zip(right_answers, answer) if ra == a)
#     if score >= 6:  # 6 es el umbral de aprobación
#         exam["passed"] = "APR"
#     else:
#         exam["passed"] = "NO APR"

#     exams.append(exam)

# results_to_screen(exams)
# results_to_csv(directory, exams)
# results_to_txt(directory, exams)

#---------------
# Notas
#---------------
# la función cv2.connectedComponentsWithStats en OpenCV generalmente espera imágenes binarias 
# donde el fondo sea negro (valor 0) y los objetos o componentes sean blancos (valor 255).
# inverted_image = cv2.bitwise_not(image)
#---------------
# inverted_image = 255 - image


