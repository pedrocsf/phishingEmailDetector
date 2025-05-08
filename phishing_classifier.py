from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import torch

# --- 1. Carregar o dataset do CSV ---
dataset = load_dataset("csv", data_files="phishing.csv")

# --- 2. Tokenizar ---
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# --- 3. Carregar modelo ---
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# --- 4. Configurar treinamento ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    no_cuda=True,  # Forçar CPU
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# --- 5. Treinar ---
trainer.train()

# --- 6. Fazer predições com o modelo treinado ---
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Teste com frases manuais
#frases = [
#    "Click here to reset your password",
#    "Obrigado por comparecer à reunião",
#    "Sua conta foi comprometida! Atualize agora"
#]

#for frase in frases:
#    resultado = classifier(frase)[0]
#    label = "Phishing" if resultado["label"] == "LABEL_1" else "Seguro"
#    print(f'"{frase}" -> {label} (confiança: {resultado["score"]:.2f})')

input = input(">>Please enter a test sentence:")
resultado = classifier(input)[0]
label= "Phishing" if resultado["label"]=="LABEL_1" else "Safe"
print(f'"{input}"-> {label} (Confidence: {resultado["score"]:.2f})')
