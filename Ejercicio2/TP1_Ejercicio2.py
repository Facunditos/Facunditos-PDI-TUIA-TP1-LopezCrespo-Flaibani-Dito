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

'''
Muestra imágenes por pantalla
'''
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
Detecta, separa y analiza los campos del encabezado del examen
'''
def analyze_header(img_enc: np.array) -> tuple:
    img_enc_rows = np.sum(img_enc,1)
    max_q_pix = np.max(img_enc_rows)
    # Se identifica en cuál índice horizontal se encuentra las líneas de los campos
    idx_lin_campos_enc_h = np.argwhere(img_enc_rows==max_q_pix)[0,0]
    lin_campos_enc_h = img_enc[idx_lin_campos_enc_h,:]
    # Se identifican los respectivos índices verticales
    y = np.diff(lin_campos_enc_h)
    idxs_lin_campos_enc_v = np.argwhere(y)
    ii = np.arange(0,len(idxs_lin_campos_enc_v),2)    # 0 2 4 ... X --> X es el último nro par antes de len(renglones_indxs)
    idxs_lin_campos_enc_v[ii]+=1
    idxs_lin_campos_enc_v = np.reshape(idxs_lin_campos_enc_v, (-1,2))
    # Se define la estructura de datos que va a contener toda la info relativa al campo de los encabezados
    campos_enc = []
    nombes_campos_enc = ['Name','Date','Class']
    for i,nombre_campo in enumerate(nombes_campos_enc):
        # Se crea la estructura de datos a utlizar para guardar la información relevante de cada campo
        datos_campo = {}
        datos_campo['nombre'] = nombre_campo
        idx_ini_lin_campos_enc_v = idxs_lin_campos_enc_v[i,0] #acaaa
        idx_fin_lin_campos_enc_v = idxs_lin_campos_enc_v[i,1]
        # Cropping del campo del encabezado
        img_campo = img_enc[:idx_lin_campos_enc_h,idx_ini_lin_campos_enc_v:idx_fin_lin_campos_enc_v]
        # img_Campo el tipo de dato para poder utilizar la función de cv2 que no acepta datos booleanos
        img_campo = img_campo.astype('uint8')
        datos_campo['img'] = img_campo
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_campo, connectivity, cv2.CV_32S)
        # Se excluye al elemento encabezado para quedarnos únicamente con los caracteres
        estadisticas_caracteres = stats[1:,:]
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
            # Se corrobora si el alumno completó correctamente el nombre_campo
            respuesta = 'OK' if (num_car <= 25 and num_palabras >= 2) else 'MAL'
            campo_enc['respuesta'] = respuesta
        elif nombre_campo == 'Date':
            # Se corrobora si el alumno completó correctamente la fecha
            respuesta = 'OK' if (num_car == 8 and num_palabras == 1) else 'MAL'
            campo_enc['respuesta'] = respuesta
        elif nombre_campo == 'Class':
            # Se corrobora si el alumno completó correctamente la clase
            respuesta = 'OK' if (num_car == 1) else 'MAL'
            campo_enc['respuesta'] = respuesta
    return campos_enc[0]['img'],campos_enc[0]['respuesta'],campos_enc[1]['respuesta'],campos_enc[2]['respuesta']

'''
Detecta y separa las zonas de cada pregunta en la imagen
'''
def identify_questions(image: np.ndarray) -> tuple:
    th = image.max() * 0.9
    # image_binary_header = cv2.threshold(image, th, 255, cv2.THRESH_BINARY)[1]
    image_binary_header = (image > th).astype(np.uint8) * 255
    image_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(image, 100, 170, apertureSize=3)
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
    header = 255 - image_binary_header[0:lines_h_sorted[0][0][1] + 5, 0:image.shape[0]]
    header_coords = (0,lines_h_sorted[0][0][1] + 5, 0,image.shape[0])
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
Dentro de cada pregunta del examen, detecta y separa las zonas de cada una de las respuestas
'''
def identify_lines(questions: list) -> tuple:
    lines_answer = []
    lines_answer_coords = []
    for question in questions:
        # Se identifican las componentes conectadas
        connectivity = 8
        # Si la imagen no está invertida (fondo negro) no funciona: 255-question
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-question, connectivity, cv2.CV_32S)
        # Se itera sobre las estadísticas para identificar los componentes con área en el rango deseado
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
def identify_answers(lines_answer: list) -> list:
    answers = []
    for line_answer in lines_answer:
        letter = ""
        #--------------------------------
        # Verificar que haya solo una letra: num_labels == 2 (fondo de la imagen + letra)
        # Componentes conectados
        #--------------------------------
        connectivity = 8
        # Si la imagen no está invertida (fondo negro) no funciona: 255-line_answer
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-line_answer, connectivity, cv2.CV_32S)
        if num_labels == 2:
            #--------------------------------
            # Segmentar las letras
            # Obtención de contornos
            # C contorno sin hijos
            # B contorno con 2 hijos
            # A contorno con 1 hijo / relación area_hijo/area_padre < 0.65
            # D contorno con 1 hijo / relación area_hijo/area_padre > 0.65
            #--------------------------------
            # Si la imagen no está invertida (fondo negro) no funciona: 255-line_answer
            contours, hierarchy = cv2.findContours(255-line_answer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            next, previuos, first_child, father = hierarchy[0][0]
            # Si first_child es -1, no tiene hijos; si es distinto de -1, tiene un hijo.
            child_count = 0
            child_area=0
            father_area=0
            aspect_ratio=0
            if first_child != -1:
                # print(f"Contorno 0 tiene un hijo con índice {first_child}")
                # Se cuentan los hijos que tiene contours[0]
                child = first_child
                while child != -1:
                    child_count += 1
                    next_child = hierarchy[0][child][0]  # Siguiente contorno en el mismo nivel
                    child = next_child
                if child_count==2: letter="B"
                elif child_count==1:
                    # Se diferencia entre "A" y "D"
                    x, y, w, h = cv2.boundingRect(contours[first_child])    # Obtener el rectángulo delimitador del contorno hijo
                    aspect_ratio = w / h                                    # Relación de aspecto
                    child_area = cv2.contourArea(contours[first_child])     # Área del contorno hijo
                    father_area = cv2.contourArea(contours[0])              # Área del contorno padre
                    if child_area / father_area < 0.65:                     # and aspect_ratio < 1:
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
        # Se recorren las respuestas y se muestra si son OK o MAL
        for idx, (key, value) in enumerate(exam["answers"].items(), 1):
            print(f"Pregunta {idx}: {value['state']}")
        print(f"Resultado: {exam['passed']}")

'''
Guarda los exámenes corregidos en un archivo CSV
'''
def results_to_csv(path: str, exams: list) -> None:
    output_csv = os.path.join(path, 'resultados_examenes.csv')
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Nombre", "Fecha", "Clase", "Pregunta", "Respuesta", "Estado", "Resultado"])
        for exam in exams:
            for idx, (key, value) in enumerate(exam["answers"].items(), 1):
                if key != 'img_name': writer.writerow([exam["id"], exam["name"], exam["date"], exam["class"], idx, value["answer"], value["state"], exam["passed"]])

'''
Guarda los exámenes corregidos en un archivo TXT
'''
def results_to_txt(path: str, exams: list) -> None:
    output_file = os.path.join(path, 'resultados_examenes.txt')
    with open(output_file, 'w') as f:
        for exam in exams:
            f.write(f"\nId: {exam['id']}\n")
            f.write(f"\nNombre: {exam['name']}\n")
            f.write(f"Fecha: {exam['date']}\n")
            f.write(f"Clase: {exam['class']}\n")
            f.write("Respuestas:\n")
            # Se recorren las respuestas y se muestra si son OK o MAL
            for idx, (key, value) in enumerate(exam["answers"].items(), 1):
                f.write(f"Pregunta {idx}: {value['state']}\n")
            f.write(f"Resultado: {exam['passed']}\n")

'''
Muestra una imagen que contiene una subimagen por examen evaluado para mostrar si el alumno aprobó 
'''
def results_to_img(path:str, exams:list) -> None:
    plt.figure()
    N_rows = (len(exams) //2) +1
    for i,examen in enumerate(exams):
        if (i==0):
            ax = plt.subplot(N_rows,2,i+1)
            imshow(exams[i]['img_name'],new_fig=False, title=exams[i]['passed'],colorbar=False, vmax=1)
        plt.subplot(N_rows,2,i+1, sharex=ax, sharey=ax), imshow(exams[i]['img_name'], new_fig=False, title=exams[i]['passed'],colorbar=False, vmax=1)
    plt.suptitle("Resultado de la evaluación")
    plt.show(block=False)
    plt.savefig(path + "/Informe.jpg", format='jpg', dpi=300, bbox_inches='tight')
    # plt.close()

#-------------------
# Programa principal
#-------------------
def main():
    print("*** Corrección de exámenes ***")
    dir_path = input("\nIngrese la carpeta que contiene los exámenes a corregir ('Tests'): ")
    # Verificar si el directorio existe
    if not os.path.exists(dir_path):
        print(f"Error: El directorio '{dir_path}' no existe.")
        return  # Terminar el programa si el directorio no se encuentra
    # Variable para verificar si se encontraron archivos PNG
    found_png_files = False
    right_answers = ["C", "B", "A", "D", "B", "B", "A", "B", "D", "D"]
    exams = []
    # Analizar cada examen
    for file in os.listdir(dir_path):
        if file.endswith('.png'):  # Solo procesar archivos PNG
            image = cv2.imread(os.path.join(dir_path, file),cv2.IMREAD_GRAYSCALE)
            if image is not None:
                found_png_files = True
                id = file
                # Procesar encabezado y preguntas
                header, _, questions, _ = identify_questions(image)
                img_name, name, date, class_n = analyze_header(header)
                # Identificar las respuestas
                lines_answer, _ = identify_lines(questions)
                answer = identify_answers(lines_answer)
                # Crear un nuevo diccionario para el examen actual
                exam = {
                    "id": id,
                    "name": name,
                    "date": date,
                    "img_name": img_name,
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
    results_to_img(dir_path, exams)
    input("Presione Enter para continuar...")

if __name__ == "__main__":
    main()