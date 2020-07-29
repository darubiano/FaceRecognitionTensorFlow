# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:48:23 2020
# mobilenet entrenado para detecion de rostros
https://github.com/yeephycho/tensorflow-face-detection
@author: darub
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path_knows = 'knows'
path_unknows = 'unknows'
path_results = 'results'

# leer mobilenet_graph_face.pb arquitectura del modelo ya entrenado
with tf.io.gfile.GFile('mobilenet_graph_face.pb', 'rb') as f:
    # leer contenido del archivo
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Grafo del modelo
with tf.Graph().as_default() as mobilenet:
    tf.import_graph_def(graph_def, name='')
print(mobilenet)

# funcion Cargar imagen
def load_image(path, name):
    # Cargar la imagen en formato RGB
    return cv2.cvtColor(cv2.imread(f'{path}/{name}'), cv2.COLOR_BGR2RGB)

# funcion para detectar la imagen con score mayor a 70%
def detect_faces(image, score=0.7):
    global boxes, scores
    (imh, imw) = image.shape[:-1]
    # agrear una dimension que necesita la entrada del modelo
    img = np.expand_dims(image, axis=0)
    
    # Inicializar mobilenet
    sess = tf.compat.v1.Session(graph=mobilenet)
    imagen_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
    boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
    scores = mobilenet.get_tensor_by_name('detection_scores:0')
    
    # Prediccion
    (boxes, scores) = sess.run([boxes,scores], feed_dict={imagen_tensor:img})
    
    # Reajustar tamaños boxes, scores
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    
    # Seleccionar puntajes altos
    idx = np.where(scores>=score)[0]
    
    # Crear bounding boxes
    bboxes = []
    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index,:]
        (left, right, top, bottom) = (xmin*imw, xmax*imw, ymin*imh, ymax*imh)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        bboxes.append([left,right,top,bottom])
        
    return bboxes

# funcion dibujar bounding boxes
def draw_box(image, box, color, line=6):
    if box==[]:
        return image
    else:
        cv2.rectangle(image, (box[0],box[2]), (box[1],box[3]), color, line)
        return image

# cagar imagen
imagen = load_image(path_unknows, 'bus.jpg')
# detectar caras
bboxes = detect_faces(imagen)
# dibujar rostros
for box in bboxes:
    detected_faces = draw_box(imagen, box,(0,255,0))
fig = plt.figure(figsize=(10,10))
plt.imshow(detected_faces)

# Extraer rostros
def extract_faces(image,bboxes,new_size=(160,160)):
    #lista de rostros
    list_faces = []
    for box in bboxes:
        left, right, top, bottom = box
        # rostros
        face = image[top:bottom,left:right]
        #guardar rostros redimensionados
        list_faces.append(cv2.resize(face, dsize=new_size))
    return list_faces
    
faces = extract_faces(imagen,bboxes)
for face in faces:
    plt.imshow(face)
    plt.show()

# FACENET
facenet = load_model('facenet_keras.h5')
#facenet.summary()
# tamaño de dato de entrada
print(facenet.input_shape)
# tamaño de dato de salida
print(facenet.output_shape)

# funcion aplicando el modelo
def compute_embedding(model,face):
    # normalizar la imagen
    face = face.astype('float32')
    # Sacar la media y  desviacion estandar
    mean,std = face.mean(), face.std()
    # normaliza los valores numericos de la imagen
    face = (face-mean)/std
    face = np.expand_dims(face, axis=0)
    # predecir vector del rostro ingresado
    embedding = model.predict(face)
    return embedding

embedding = compute_embedding(facenet,faces[0])
# Caracteristicas detectadas del rostro 
print(embedding)

# Embeddings de referencia
know_embeddings = []
print('Procesando rostros conocidos')
for name in os.listdir(path_knows):
    if name.endswith('.jpg'):
        print(f'{name}')
        image = load_image(path_knows, name)
        bboxes = detect_faces(image)
        face = extract_faces(image, bboxes)
        know_embeddings.append(compute_embedding(facenet, face[0]))
        
# lista de rostros procesados
print(len(know_embeddings))

# Comparacion de rostros
# embs_ref de referencia, emb_desc desconocidos, distancia de 11
def compare_faces(embs_ref, emb_desc, umbral=11):
    distancias = []
    for emb_ref in embs_ref:
        # restar los dos vectores
        distancias.append(np.linalg.norm(embs_ref-emb_desc))
        #print(f'umbral: {np.linalg.norm(embs_ref-emb_desc)}')
    distancias = np.array(distancias)
    return distancias, list(distancias<=umbral)

# Reconocimiento de rostros 
print('Procesando imágenes desconocidas...')
for name in os.listdir(path_unknows):
    if name.endswith('.jpg'):
        print(f'   {name}')
        image = load_image(path_unknows,name)
        bboxes = detect_faces(image)
        faces = extract_faces(image,bboxes)
        
        # Por cada rostro calcular embedding
        img_with_boxes = image.copy()
        for face, box in zip(faces,bboxes):
            emb = compute_embedding(facenet,face)
            
            _, reconocimiento = compare_faces(know_embeddings,emb,20)
            
            if any(reconocimiento):
                print('match!')
                img_with_boxes = draw_box(img_with_boxes,box,(0,255,0))
            else:
                img_with_boxes = draw_box(img_with_boxes,box,(255,0,0))
            
        cv2.imwrite(f'{path_results}/{name}',cv2.cvtColor(img_with_boxes,cv2.COLOR_RGB2BGR))
print('Fin')




 

























