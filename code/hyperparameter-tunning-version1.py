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
alvo_treino, caracteristicas_treino = importImage('data/chest_xray/train/')

# importar e preprocessar base de teste
alvo_teste, caracteristicas_teste = importImage('data/chest_xray/test/')

#========================================
# rede neural artificial com keras HYPER
#========================================

from tensorflow import keras
from kerastuner import HyperModel

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        
        model.add(keras.layers.Dense(
                units= hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                input_shape=self.input_shape,
                activation='sigmoid')
        )

        model.add( 
            keras.layers.Dense(
                    units=hp.Int(
                        'units',
                        min_value=32,
                        max_value=512,
                        step=32,
                        default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            keras.layers.Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            keras.layers.Dense(
                self.num_classes, 
                activation='softmax'))

        model.compile(
            loss="sparse_categorical_crossentropy", 
            optimizer="adam", 
            metrics=["accuracy"])

        return model


NUM_CLASSES = 2  # cifar10 number of classes
INPUT_SHAPE = tuple([caracteristicas_treino.shape[1]])  # cifar10 images input shape

hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)


from kerastuner.tuners import Hyperband

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2

tuner = Hyperband(
    hypermodel,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='val_acc',
    seed=1,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='paper-neural-networks',
    project_name='proj'
)

tuner.search_space_summary()

N_EPOCH_SEARCH = 30

tuner.search(caracteristicas_treino, alvo_treino, epochs=N_EPOCH_SEARCH, validation_split=0.1)

# Show a summary of the search
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model.
loss, accuracy = best_model.evaluate(caracteristicas_teste, alvo_teste)