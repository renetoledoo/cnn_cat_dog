import os

DIRETORIO_PAI = r"C:\Users\rbtol\Downloads\dogs-vs-cats\train\train"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DIRETORIO_GATO = os.path.join(BASE_DIR, "train", "cat")
DIRETORIO_CACHORRO = os.path.join(BASE_DIR, "train" ,"dog")

os.makedirs(DIRETORIO_GATO, exist_ok=True)
os.makedirs(DIRETORIO_CACHORRO, exist_ok=True)

lista_arquivos = os.listdir(DIRETORIO_PAI)

for nome_arquivo in lista_arquivos:
    caminho_atual = os.path.join(DIRETORIO_PAI, nome_arquivo)

    if "dog" in nome_arquivo:
        destino = os.path.join(DIRETORIO_CACHORRO, nome_arquivo)
    else:
        destino = os.path.join(DIRETORIO_GATO, nome_arquivo)

    os.rename(caminho_atual, destino)



print("Finalizado")