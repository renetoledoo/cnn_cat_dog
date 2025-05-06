import os
import numpy as np
import keras
from modelo import build_engine, criar_gerador_teste
import cv2

CONTEXTO =  os.path.dirname(os.path.abspath(__file__))
IMAGE_WIDTH = IMAGE_HEIGHT = 150


def classificar_imagem(model, imagem_diretorio):
    imagem = cv2.imread(imagem_diretorio)

    if imagem is None:
        print(f"Erro ao carregar a imagem: {imagem_diretorio}")
        return

    imagem = cv2.resize(imagem, (IMAGE_WIDTH, IMAGE_HEIGHT))
    imagem = imagem.astype('float32') / 255.0
    imagem = imagem.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    previsao = model.predict(imagem)
    previsao = previsao[0][0]
    return ('Cachorro' if previsao > 0.5 else 'Gato', previsao)

       


if __name__ == '__main__':
    caminho_model =  os.path.join(CONTEXTO, 'model3.keras')
    if not os.path.exists(caminho_model): 
        build_engine()
    else:
        model = keras.models.load_model("model3.keras")
        caminho_imagens = os.path.join('varios_gatos.png')
        retorno = classificar_imagem(model, caminho_imagens)
        print(retorno)
        # criar_gerador_teste(model)
        # criar_gerador_teste(model)
       
