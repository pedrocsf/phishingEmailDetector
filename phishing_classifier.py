import os
import time
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

# Caminhos para os arquivos
csv_file = "phishing.csv"
last_modified_file = "last_modified.txt"

# Função para verificar se o arquivo foi modificado
def is_file_modified(file_path, last_modified_file):
    try:
        # Verifica a data de modificação do arquivo CSV
        current_mod_time = os.path.getmtime(file_path)
        
        # Lê o timestamp da última modificação registrada
        if os.path.exists(last_modified_file):
            with open(last_modified_file, "r") as f:
                last_mod_time = float(f.read().strip())
        else:
            last_mod_time = 0  # Se o arquivo não existir, significa que nunca foi registrado

        # Compara as datas
        if current_mod_time > last_mod_time:
            # Atualiza o arquivo com o timestamp atual
            with open(last_modified_file, "w") as f:
                f.write(str(current_mod_time))
            return True  # O arquivo foi modificado
        return False  # O arquivo não foi modificado
    except Exception as e:
        print(f"Erro ao verificar o arquivo: {e}")
        return False

# Função para treinar o modelo
def train_model():
    print("Iniciando o treinamento...")

    # Carregar o dataset
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['text', 'label'])  # Remover valores nulos, se houver

    # Carregar o tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Tokenize os dados
    train_encodings = tokenizer(list(df['text']), truncation=True, padding=True)
    train_labels = list(df['label'])

    # Criar o modelo
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Configurar os parâmetros de treinamento
    training_args = TrainingArguments(
        output_dir="./results",          # Diretório de saída
        num_train_epochs=3,              # Número de épocas
        per_device_train_batch_size=8,   # Tamanho do lote
        warmup_steps=500,                # Passos para o warmup
        weight_decay=0.01,               # Decaimento de peso
        logging_dir="./logs",            # Diretório de logs
        no_cuda=True,                    # Usar apenas CPU
    )

    # Criar o Trainer
    trainer = Trainer(
        model=model,                         # O modelo
        args=training_args,                  # Argumentos de treinamento
        train_dataset=(train_encodings, train_labels),  # Dados de treinamento
    )

    # Treinar o modelo
    trainer.train()

    return model, tokenizer

# Função para classificar uma frase de phishing
def classify_phrase(model, tokenizer):
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    user_input = input("Digite uma frase para verificar se é phishing: ")
    prediction = classifier(user_input)
    print(f"Predição: {prediction}")

# Verificar se o arquivo foi alterado
if is_file_modified(csv_file, last_modified_file):
    # Se o arquivo foi modificado, treinar o modelo novamente
    model, tokenizer = train_model()
else:
    # Se o arquivo não foi modificado, carregar o modelo treinado previamente
    print("O arquivo não foi modificado. Usando o modelo treinado previamente.")
    model = DistilBertForSequenceClassification.from_pretrained("./results")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Realizar a classificação com o modelo treinado
classify_phrase(model, tokenizer)
