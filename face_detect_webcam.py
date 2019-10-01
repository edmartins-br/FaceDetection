import cv2

video = cv2.VideoCapture(0)

while True:
    conectado, frame = video.read()

    cv2.imshow('Video', frame)

    # O parametro do WaiKey quando for zero, espera ser digitada qualquer tecla
    # Caso passe o valor 1, o programa espera por uma tecla específica.
    #É importante inserir o wwaitKey pois se não for inserido, o frame abre e já fecha automaticamente
    # não ficando disponível na tela
    if cv2.waitKey(1) == ord('q'):
        break
video.release() #Comando necessario para liberar a memoria
cv2.destroyAllWindows()

image_path = 'time.jpg' #imagem
cascade_path = 'haarcascade_frontalface_default.xml'# arquivo de cascade / modelo

#criando o classidicador
clf = cv2.CascadeClassifier(cascade_path)

img = cv2. imread(image_path) #ler a imagem
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# converter para preto e branco / grayscale

#Primeiro parametro é a escala e o segundo é o tamanho minimo da imagem em PX
faces = clf.detectMultiScale(gray, 1.2, 10)

#Desenha os retangulos na imagem
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()