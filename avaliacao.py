import matplotlib.pyplot as plt
import tensorflow as tf





def resultados_graficos(history):
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(12, 5))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Treinamento')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Treinamento')
    plt.plot(epochs_range, history.history['val_loss'], label='Validação')
    plt.title('Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_dataset_predictions(dataset_test, model):
    class_names = list(dataset_test.class_indices.keys())
    
    features, labels = next(dataset_test)  

    predictions = model.predict(features).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Labels:      %s' % labels)
    print('Predictions: %s' % predictions.numpy())

    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(features[i])
        plt.title(class_names[int(predictions[i])])
    plt.show()
    
    
        
# def classificar_lote(model):
#     resultado = {
#         'imagem': [],
#         'classificacao': [],
#         'previsao': [],
#         'foto': []
#     }

#     for i in range(40):  
#         nome_arquivo = f'{i}.jpg'
#         caminho_imagens = os.path.join('test', nome_arquivo)

#         imagem = cv2.imread(caminho_imagens)
#         if imagem is None:
#             continue
#         tipo, previsao = classificar_imagem(model, caminho_imagens)
#         # Preenche o dicionário
#         resultado['imagem'].append(caminho_imagens)
#         resultado['classificacao'].append(tipo)
#         resultado['previsao'].append(float(previsao))
#     criar_planilha(resultado)
#     return resultado