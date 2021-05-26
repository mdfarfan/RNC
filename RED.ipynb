#%% import the necessary packages
import numpy as np
import logging, os
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
import os
import tensorflow as tf
from tensorflow import keras
#%%RED 1
"RED 1____________________________________________________________________"

"""# Inicializar la CNN"""

classifier = Sequential()

"""# Paso 1 - Convolución y Maxpooling"""

#Capa 1
classifier.add(Conv2D(filters = 64,kernel_size = (3, 3),input_shape = (128,128, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Capa 2
classifier.add(Conv2D(filters = 64,kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

"""# Paso 3 - Flattening"""

classifier.add(Flatten())

"""# Paso 4 - Full Connection"""

#Capa 1
classifier.add(Dense(units = 256, activation = "relu"))
classifier.add(Dropout(0.5))

#Capa 2
classifier.add(Dense(units = 256, activation = "relu"))
classifier.add(Dropout(0.5))

#Salida
classifier.add(Dense(units = 1, activation = "sigmoid"))

"""# Compilar la CNN"""

classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])
batch = 32

#Guardamos modelo
classifier.save('modelos_entrenados/RED1.h5') 
#%%Cargar modelo
classifier = tf.keras.models.load_model('modelos_entrenados/RED1.h5')

# Check its architecture
classifier.summary()
"""____________________________________________________________________________"""
#%%Filtros
"Mostrar los filtros"
layer = classifier.layers #Obtenemos las capas 
filters,biases = classifier.layers[2].get_weights()
print(layer[2].name,filters.shape)

#Creamos la imagen para mostrar
fig1 = plt.figure(figsize=(8,12))
columnas = 8
filas = 8
nfiltros = columnas*filas

#For para graficar todos los filtros
for i in range(1,nfiltros+1):
    f = filters[:,:,:,i-1]
    fig1 = plt.subplot(filas,columnas,i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:,:,0], cmap = 'gray')
    
plt.show()
"""____________________________________________________________________________"""

#%%Hacer predicciones red 1
"Mostrar los salidas red 1________________________________________________________________"
#Mostrar las imagenes de salida de las convoluciones
conv_layer = [0,2] #indices de las capas convolucionales
outputs = [classifier.layers[i].output for i in conv_layer] #Guardamos las salidas
model_short = Model(inputs = classifier.inputs, outputs = outputs) #Creamos nuevo modelo
print(model_short.summary()) 

# disable the warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#Ruta de los archivos
image_path = "ambos"

images = []
    
# load all images into a list
for img in os.listdir(image_path):
        img = os.path.join(image_path, img)
        img = image.load_img(img, target_size=(128,128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # normalize the image
        processed_image = np.array(img, dtype="float") / 255.0
        images.append(processed_image)
        
#Lista de nuestras imagenes
images = np.vstack(images)


mn = len(images)
charac = []
feat = np.zeros( (1, 61, 61, 64) )

# Hacer multiples predicciones
for i in range(mn):  
    img = np.zeros( (1, 128, 128, 3) )  
    #Pasamos una imagen
    img[0,:,:,:] = images[i,:,:,:]
    #Hacemos la predicción
    pred_result = model_short.predict(img)
    #Extraemos las imagenes de la segunda capa
    feat = pred_result[1]
    #Añadimos la predicción a nuestro array
    charac.append(feat)

#%%RED 2
"Segunda red________________________________________________________________________________"
"""# Inicializar la CNN"""

red2 = Sequential()

"""# Paso 1 - Convolución y Maxpooling"""

#Capa 1
red2.add(Conv2D(filters = 32,kernel_size = (3, 3),input_shape = (128,128, 3), activation = "relu"))
red2.add(MaxPooling2D(pool_size = (2,2)))
#Capa 2
red2.add(Conv2D(filters = 64,kernel_size = (3, 3), activation = "relu"))
red2.add(MaxPooling2D(pool_size = (2,2)))
#Capa 3
red2.add(Conv2D(filters = 128,kernel_size = (3, 3),activation = "relu"))
red2.add(MaxPooling2D(pool_size = (2,2)))


"""# Paso 3 - Flattening"""

red2.add(Flatten())

"""# Paso 4 - Full Connection"""

#Capa 1
red2.add(Dense(units = 128, activation = "relu"))
red2.add(Dropout(0.5))

#Capa 2
red2.add(Dense(units = 128, activation = "relu"))
red2.add(Dropout(0.5))

#Salida
red2.add(Dense(units = 1, activation = "sigmoid"))

"""# Compilar la CNN"""

red2.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])
batch = 32

#Guardamos modelo
classifier.save('modelos_entrenados/RED2.h5') 


#%%Cargar modelo

red2 = tf.keras.models.load_model('modelos_entrenados/RED2.h5')
# Check its architecture
red2.summary()

#%%Mostar filtros red 2
"""_____________________________________________________________________________"""
"Mostrar los filtros"
layer2 = classifier.layers #Obtenemos las capas 
filters2,biases = classifier.layers[2].get_weights()
print(layer[2].name,filters2.shape)

#Creamos la imagen para mostrar
fig1 = plt.figure(figsize=(8,12))
columnas = 8
filas = 8
nfiltros = columnas*filas

#For para graficar todos los filtros
for i in range(1,nfiltros+1):
    f = filters2[:,:,:,i-1]
    fig1 = plt.subplot(filas,columnas,i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:,:,0], cmap = 'gray')
    
plt.show()
"""_________________________________________________________________________________"""

#%%Hacer predicciones red 2
"Mostrar los salidas red 2________________________________________________________________"
#Mostrar las imagenes de salida de las convoluciones
conv_layer2 = [0,2] #indices de las capas convolucionales
outputs2 = [red2.layers[i].output for i in conv_layer2] #Guardamos las salidas
print(outputs2)
model_short2 = Model(inputs = red2.inputs, outputs = outputs2) #Creamos nuevo modelo
print(model_short2.summary()) 

# disable the warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#Ruta de los archivos
image_path2 = "ambos"

images2 = []
    
# load all images into a list
for img2 in os.listdir(image_path2):
        img2 = os.path.join(image_path2, img2)
        img2 = image.load_img(img2, target_size=(128,128))
        img2 = image.img_to_array(img2)
        img2 = np.expand_dims(img2, axis=0)
        # normalize the image
        processed_image = np.array(img2, dtype="float") / 255.0
        images2.append(processed_image)

#Lista de nuestras imagenes
images2 = np.vstack(images2)

mn2 = len(images)
charac2 = []
feat2 = np.zeros( (1, 61, 61, 64) )

# Hacer multiples predicciones
for i in range(mn2):  
    img2 = np.zeros( (1, 128, 128, 3) )  
    #Pasamos una imagen
    img2[0,:,:,:] = images2[i,:,:,:]
    #Hacemos la predicción
    pred_result2 = model_short2.predict(img2)
    #Extraemos las imagenes de la segunda capa
    feat2 = pred_result2[1]
    #Añadimos la predicción a nuestro array
    charac2.append(feat2)



#%%Distancias euclidianas
"___________________________________________________________________________"

"Calcular la distancia euclidiana"
disteu = np.zeros( (64,64) )
distancias1 = np.zeros( (2000, 64, 64) )
from scipy.spatial import distance
"""Las columnas (i) guardan la imagen correspondiente comparada con las... 
 otras 64"""

for k in range (2000):
    for j in range (64):
        for i in range (64):
            x = charac[i]
            x1 = x.reshape(-1)
            y =  charac[i]
            y1 = y.reshape(-1)
            dis = distance.euclidean(x1,y1)
            disteu[j][i] = dis
    distancias1[k][:][:] = disteu
    #j son filas, i columnas
    
"_____________________________________________________________________________"
"Valor minimo y máximo de las distancias"
valmin = distancias1.min()
valmax = distancias1.max() 
vact = 0
newmat = np.zeros( (64,64) )
distnorm = np.zeros( (2000, 64, 64) )
from sklearn.preprocessing import StandardScaler
escala_x = StandardScaler()   
for k in range (2000):
    for j in range (64):
        for i in range (64):
            vact = distancias1[k][i][j]
            vact = (vact *255)/valmax
            distnorm[k][i][j] = vact
            
#%%Sacar promedio
valact = 0
valt = 0
mataux = np.zeros( (1, 64, 64) )#Matriz para llenar cada ciclo
#Sumar todos los datos
for k in range (2000):
    for j in range (64):
        for i in range (64):
            valact = distnorm[k][i][j]
            valt = mataux[0][i][j]
            valor = valt + valact
            mataux[0][i][j] = valor

#Dividir los datos para obtener promedio
for j in range (64):
        for i in range (64):
            valact = mataux[0][i][j]
            valact = valact/20
            mataux[0][i][j] = valact
print(mataux)

#%%Graficar

"Graficar"
import matplotlib.pyplot as plt

for i in range(10):   
    plt.figure()
    plt.title('Mapa {}'.format(i+1))
    newmat = distnorm[i][:][:]
    plt.imshow(newmat,cmap = 'jet', vmin=0, vmax=255)
    plt.colorbar()


plt.figure()
plt.imshow(feat[6, :, :, 60], cmap='gray')
plt.figure()
plt.imshow(feat2[6, :, :, 54], cmap='gray')


columns = 8
rows = 8
#For para graficar las imagenes
for i in range(1, columns*rows +1):
    fig = plt.subplot(rows, columns, i)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(feat[6, :, :, i-1], cmap='gray')
plt.show()
for i in range(1, columns*rows +1):
    fig = plt.subplot(rows, columns, i)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(feat2[6, :, :, i-1], cmap='gray')
plt.show()

 
z1 = distnorm[0:5][:][:]
data = z1.reshape(-1)
plt.title('Histograma Gatos')
plt.hist(data, bins = 50)
plt.show()

z1 = distnorm[6:10][:][:]
data = z1.reshape(-1)
plt.title('Histograma Perros')
plt.hist(data, bins = 50)
plt.show()

#Filtros en numero
fltss = np.zeros((64,3, 3))  
#Pasar los valores de los filtros a lista
for k in range (64):
    xy = filters[:, :, :, k]
    filtvalue = xy.reshape(-1)
flts = np.array_split(filtvalue, 64)
#Convertir las listas en matrices individuales
for i in range (64):
    w = flts[i]
    w = w.reshape(3,3)
    fltss[i][:][:] = w
#%%