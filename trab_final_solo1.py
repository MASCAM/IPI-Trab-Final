from cv2 import cv2                     #para abrir, escrever e manipular a imagem
import math                             #uso de funções matemáticas
import numpy as np                      #manipulação de arrays
from matplotlib import pyplot as plt    #plotagem de gráficos e imagens
from PIL import Image

def mmhdome_or_basin(img, value, dome = True):  #faz o topo de contraste se verdadeiro ou base se for falso
    
    for x in range(0, len(img)):        #percorrendo-a
        for y in range(0, len(img[0])):
            pixels = []
            if (x != 0):
                a = img[x - 1][y]
                pixels.append(a)            #armazenando os pixels vizinhos de 4
                                            
            if (x + 1 < len(img)):
                b = img[x + 1][y]
                pixels.append(b)

            if (y + 1 < len(img[0])):
                c = img[x][y + 1]
                pixels.append(c)

            if (y > 0):
                d = img[x][y - 1]
                pixels.append(d)

        mean = np.mean(pixels)                      #faz a média dos vizinhos
        if (dome):                                  #se ajusta o contraste por cima
            if (abs(img[x][y] - mean) > value):     #se o contraste for maior que o valor estipulado
                if (img[x][y] > mean):              #ajusta o valor do nível de brilho
                    img[x][y] = mean + value

                else:
                    img[x][y] = mean - value

        else:                                       #se ajusta o contraste por baixo
            if (abs(img[x][y] - mean) < value):     #se o contraste for menor que o valor estipulado
                if (img[x][y] > mean):              #ajusta o valor do nível de brilho
                    img[x][y] = mean + value

                else:
                    img[x][y] = mean - value

    return img                          #após percorrer uma coluna retorna a coluna criada

def mmbinary(img, value):
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            if (img[x][y] > value):
                img[x][y] = 255

            else:
                img[x][y] = 0

    return img

def add_integer(img, integer):              #adiciona um inteiro a cada pixel da imagem
    img = img.astype(np.uint32)
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            img[x][y] += integer
            if (img[x][y] >= 255):          #não deixa passar do limite de brilho de 255
                img[x][y] = 255

    img = img.astype(np.uint8)
    return img

def create_kernel(n, radius):                   #cria o kernel para erosão e dilatação em formato de disco
    kernel = np.zeros((n,n), dtype = np.uint8)                    #cria a matriz de base
    center_x = math.floor(n / 2)                 #determina o centro na matriz de base
    center_y = center_x
    x, y = np.meshgrid(np.arange(kernel.shape[0]), np.arange(kernel.shape[1]))  #cria dois vetores de coordenadas na qual o valor de cada ponto é sua distância em relação ao ponto (0, 0)
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)  #cria a matriz distância onde o valor de cada ponto é sua distância em relação ao centro calculado
    kernel[np.where(distance >= radius)] = 1         #nos pontos da matriz distância onde o valor é maior ou igual ao valor do raio especificado, atribui o valor 1 ao ponto da matriz do núcleo    
    kernel[np.where(distance > radius + 0.5)] = 0    #nos pontos da matriz distância onde o valor é maior do que 0,5 + o valor do raio especificado, atribui o valor 0 ao ponto da matriz do núcleo    
    return kernel

def image_closing(img, kernel):          #faz o fechamentoda imagem e o retorna
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=6)    #gera o fechamento da imagem
    return closing                                              #o retorna

def image_opening(img, kernel):          #faz o fechamentoda imagem e o retorna
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=4)    #gera o fechamento da imagem
    return opening                                              #o retorna

def morphological_gradient(img, kernel):        #realiza o gradiente morfológico da imagem
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=2)    #gera o gradiente
    return gradient                                                 #o retorna

def find_draw_and_measure_contours(img, canny_input):
    for x in range(0, len(canny_input)):
        for y in range(0, len(canny_input[0])):
            canny_input[x][y] = 255 - canny_input[x][y]


    canny_output = cv2.Canny(canny_input, 100, 200) #acha as bordas da imagem binária
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    #acha os contornos em uma hierarquia definida
    circles = contours
    actual_color = (0, 0, 255)
    for i in range(len(contours)):
        cv2.drawContours(img, circles, i, actual_color, 2, cv2.LINE_8, hierarchy, 0)    #desenha os círculos na imagem com as cores definidas

def find_and_highlight_craters(image, kernel):
    img = cv2.imread(image, 0)                   #lê a imagem de entrada
    img = cv2.resize(img, (0,0), fx=2, fy=2)     #aumenta o tamanho da imagem de entrada
    resize =  cv2.imread(image)
    resize = cv2.resize(resize, (0,0), fx=2, fy=2)
    img = add_integer(img, 70)
    cv2.imwrite('mmadmsolo1.jpg', img)
    img = mmhdome_or_basin(img, 10, True)
    cv2.imwrite('mmhdomesolo1.jpg', img)
    img = mmhdome_or_basin(img, 20, False)
    cv2.imwrite('mmhbasinsolo1.jpg', img)
    img = morphological_gradient(img, kernel)
    cv2.imwrite('gradientsolo1.jpg', img)
    img = mmbinary(img, 32)
    cv2.imwrite('binarysolo1.jpg', img)
    img = image_closing(img, kernel)
    cv2.imwrite('closingsolo1.jpg', img)
    img = image_opening(img, kernel)
    cv2.imwrite('openingsolo1.jpg', img)
    find_draw_and_measure_contours(resize, img)
    cv2.imwrite('finalsolo1.jpg', resize)


kernel = create_kernel(7, 2) #cria o kernel com o tamanho NxN e raio especificados
find_and_highlight_craters('Solo_1.jpg', kernel)