import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch

# Caminho do dataset
csv_path = "phishing.csv"

# Verifica se o usuário quer treinar o modelo
train_choice = input("Deseja treinar o modelo novamente? (s/n): ").strip().lower()

if train_choice == 's':
    # Carrega os dados
    df = pd.read_csv(csv_path)

    # Verifica se há as colunas esperadas
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("O CSV deve conter as colunas 'text' e 'label'.")

    # Divide os dados em treino e validação
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2
    )

    # Tokenização
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Conversão para Dataset
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    # Modelo
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Configuração de treino
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        #evaluation_strategy="epoch",
        #save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        no_cuda=True  # substituto de no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Treinamento
    trainer.train()

    # Salva modelo e tokenizer
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

else:
    # Usa modelo previamente treinado
    if not os.path.exists("./trained_model"):
        raise FileNotFoundError("O modelo treinado não foi encontrado. Execute com treinamento pelo menos uma vez.")

    model = DistilBertForSequenceClassification.from_pretrained("./trained_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./trained_model")

# Pipeline de inferência
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Entrada do usuário
while True:
    text = input("\nDigite o texto do email (ou 'sair' para encerrar): ")
    if text.lower() == 'sair':
        break
    prediction = classifier(text)[0]
    label = prediction['label']
    score = prediction['score']
    print(f"Predição: {label} (confiança: {score:.2%})")
