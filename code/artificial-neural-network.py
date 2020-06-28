'''
FEATURE ENGINEERING
Redes Neurais Artificiais: Teoria e Aplicacao na Mineracao de Dados

@claudio alves monteiro
github.com/claudioalvesmonteiro
junho 2020
'''

# importar pacotes
import pandas as pd
import numpy as np
import keras 
import cv2
import matplotlib.pyplot as plt
import os

# ler imagem em escala de cinza
#img_array = cv2.imread('data/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg', cv2.IMREAD_GRAYSCALE) 

#=========================
# preprocessamento
#=========================

def importImage(caminho):
  ''' funcao importarPlantas
      caminho: caminho ate a pasta das imagens
  '''
  # acesso as subpastas
  subpasta = ['NORMAL', 'PNEUMONIA']
  # criar lista de imagens e rotulos
  imagens = []
  rotulo = []
  # para cada subpasta, importar e pre-processar imagens
  for sub in subpasta:
    arquivos_img = os.listdir(caminho+sub)
    for imagem in arquivos_img:
        img_array = cv2.imread(caminho+sub+'/'+imagem, cv2.IMREAD_GRAYSCALE) 
        try:
            img_array = cv2.resize(img_array, (60,60), interpolation = cv2.INTER_AREA) 
            img_array = keras.utils.normalize(img_array, axis=1) 
            img_array = img_array.reshape(-1)        
            imagens.append(img_array)
            if sub == 'PNEUMONIA':
                rotulo.append(1)
            else:
                rotulo.append(0)
        except:
            print(imagem)
  # retornar imagens e rotulos
  return np.asarray(rotulo), np.asarray(imagens)

# importar e preprocessar base de treinamento
rotulo, caracteristicas = importImage('data/chest_xray/train/')

# importar e preprocessar base de teste
rotulo_test, caracteristicas_test = importImage('data/chest_xray/test/')

#========================================
# rede neural artificial com keras
#========================================

# inicializa um modelo sequencial [outros modelos podem ser empregados, mas abordamos o mais simples]
model = keras.models.Sequential() 
# camada de entrada, input no mesmo shape dos dados
model.add(keras.layers.core.Dense(64, input_shape=tuple([caracteristicas.shape[1]]), activation='sigmoid'))
# camada oculta
model.add(keras.layers.core.Dense(512, activation='relu'))
model.add(keras.layers.core.Dense(64, activation='relu'))
# camada de saida (decisao)
model.add(keras.layers.core.Dense(2,  activation='softmax'))
# otimizacao
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# treinamento
model.fit(caracteristicas, rotulo, epochs=30)

# previsao
predictions = model.predict(caracteristicas_test)

# decidir por '0' se a probabilidade de 0 for maior, decidir por 1 ao contrario
rotulo_pred = [0 if x[0] > x[1] else 1 for x in predictions]

#========================================
# avaliando modelo
#========================================

# gerar matriz de confusao 
from sklearn.metrics import confusion_matrix
vp, fn, fp, vn = confusion_matrix(rotulo_test, rotulo_pred, labels=[1,0]).reshape(-1)
total = vp+fn+fp+vn
print('Verdadeiro Positivo: ', vp, '/', round(vp/total*100, 2), '%')
print('Falso Positivo: ', fp, '/', round(fp/total*100, 2), '%')
print('Verdadeiro Negativo: ', vn, '/', round(vn/total*100, 2), '%')
print('Falso Negativo: ', fn, '/', round(fn/total*100, 2), '%')
print('Acuracia: ', (vp+vn)/total, '/', round((vp+vn)/total*100, 2), '%')

#---- AUC-ROC

# selecionar probabilidade para classe 1
probs = [x[1] for x in predictions]

from sklearn.metrics import roc_curve,roc_auc_score
taxa_falso_positivo , taxa_verdadeiro_positivo , thresholds = roc_curve(rotulo_test , probs)

# visualizar ROC
def plot_roc_curve(taxa_falso_positivo, taxa_verdadeiro_positivo): 
    import matplotlib.pyplot as plt
    plt.plot(taxa_falso_positivo, taxa_verdadeiro_positivo) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('Taxa Falso Positivo') 
    plt.ylabel('Taxa Verdadeiro Positivo') 
    plt.show()    

plot_roc_curve(taxa_falso_positivo, taxa_verdadeiro_positivo) 