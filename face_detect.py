import cv2

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

