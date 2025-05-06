import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout

from avaliacao import resultados_graficos, plot_dataset_predictions
import os 

IMAGE_WIDTH = IMAGE_HEIGHT = 150
BATCH_SIZE = 32
EPOCHS = 20
CONTEXTO =  os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CONTEXTO, 'train')
DATASET_DIR_TESTE = os.path.join(CONTEXTO, 'test')

RETORNO = []


### PARA MEUS TESTES
def criar_gerador_teste(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    dataset_test = test_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    plot_dataset_predictions(dataset_test, model)

def criar_geradores(image_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    treino = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',  
        shuffle=True
    )

    validacao = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )
    # print("Classes encontradas:", treino.class_indices)
    # print("Classes encontradas:", validacao.class_indices)

    return treino, validacao



def criar_modelo():
    classificador = Sequential()
    classificador.add(Conv2D(32, (3, 3), activation='relu', input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    classificador.add(MaxPool2D((2, 2)))

    classificador.add(Conv2D(64, (3, 3), activation='relu'))
    classificador.add(MaxPool2D((2, 2)))

    classificador.add(Conv2D(128, (3, 3), activation='relu'))
    classificador.add(MaxPool2D((2, 2)))

    classificador.add(Flatten())
    classificador.add(Dense(256, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(1, activation='sigmoid'))

    classificador.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )
    
    return classificador




def criar_callbacks(model_path='model3.keras'):
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1)
    ]
    
    
    

def treinar_modelo(model, treino, validacao, callbacks, epochs):
    return model.fit(
        treino,
        epochs=epochs,
        validation_data=validacao,
        callbacks=callbacks
    )



def build_engine():
    treino_gen, val_gen = criar_geradores(image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE)
    modelo = criar_modelo()
    callbacks = criar_callbacks()
    history = treinar_modelo(modelo, treino_gen, val_gen, callbacks, epochs=EPOCHS)
    resultados_graficos(history)
    

