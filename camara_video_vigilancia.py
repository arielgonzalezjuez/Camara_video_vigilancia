import cv2

#Carga el clasificador pre-entrenado de deteccion de rostros.
face_cascade = cv2.cascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')

#Cargar el video desde la camara de videovigilancia
video_capture = cv2.VideoCapture(0)

while True:
    #Leer un frame del video
   ret, frame = video_capture.read()


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,
    minNeighbors=5, miniSize=(30,30))

    # Dibujar un rectangulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces: cv2.rectangle(frame, (x, y), (x+w, y+h),
    (0, 255, 0), 3)

    # Dibujar una cuadricula en el frame
    rows, cols, _ = frame.shape
    for i in range(0, rows, 50):
        cv2.line(frame, (0, i),
                 (cols, i), (255, 0, 0), 1)
        for j in range(0, cols, 50):
            cv2.line(frame, (j, 0), (j, rows),
                     (255, 0, 0), 1)

    # Mostrar el frame resultante
    cv2.imshow('Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitkey(1) & 0xFF == ord('q'):

    # Liberar los recursos
    video_capture.release()
    cv2.destroyAllWindows()
