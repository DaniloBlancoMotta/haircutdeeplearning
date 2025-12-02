# ==============================================================================
#                      README: CLASSIFICADOR DE TIPO DE CABELO
# ==============================================================================
"""
# 🚀 Classificador de Tipo de Cabelo (CNN PyTorch)

Este script implementa uma Rede Neural Convolucional (CNN) em PyTorch para a classificação binária de imagens de cabelo (liso vs. cacheado/crespo).

## ⚙️ Configuração
- **Dataset:** Hair Type dataset (treino e teste).
- **Arquitetura do Modelo:** CNN customizada de 3 camadas (Conv -> Linear -> Linear).
- **Otimizador:** Stochastic Gradient Descent (SGD) com momentum.
- **Reproducibilidade:** Semente (SEED=42) fixada para numpy e PyTorch.
- **Input Shape:** (3, 200, 200).

## 💡 Arquitetura da CNN
1. **Conv2d:** 32 filtros, kernel (3, 3), ReLU.
2. **MaxPool2d:** kernel (2, 2).
3. **Linear (Hidden):** 64 neurônios, ReLU.
4. **Linear (Output):** 1 neurônio, Sigmoid.

## 📈 Treinamento
O script inclui a preparação de dados, a definição do modelo e a função de treinamento. O pipeline é projetado para responder às questões do exercício, incluindo a aplicação de **Data Augmentation** nas últimas 10 épocas.
"""

# ==============================================================================
#                               CONFIGURAÇÃO INICIAL
# ==============================================================================

# Importações essenciais para Deep Learning em Python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import shutil
import requests
import zipfile

# 1. Configuração de Reproducibilidade (SEED)
# Essencial para garantir que os resultados sejam consistentes em diferentes execuções.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuração específica para CUDA (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Define determinismo para operações CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define o dispositivo de execução (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
#                               DOWNLOAD E PREPARAÇÃO DOS DADOS
# ==============================================================================

DATA_URL = "https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip"
DATA_PATH = "data.zip"
EXTRACT_DIR = "."

# Função para download e extração do dataset
def download_and_extract_data():
    """Baixa e extrai o dataset Hair Type de Kaggle."""
    if os.path.exists("./train") and os.path.isdir("./train"):
        print("Diretório 'train' já existe. Pulando download.")
        return
        
    print(f"Baixando dados de: {DATA_URL}")
    try:
        r = requests.get(DATA_URL, stream=True)
        r.raise_for_status() # Verifica se o download foi bem-sucedido
        with open(DATA_PATH, 'wb') as f:
            f.write(r.content)
        print("Download concluído. Descompactando...")
        
        with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extração concluída em ./data.")
        os.remove(DATA_PATH) # Remove o arquivo zip
    except Exception as e:
        print(f"Erro no download ou extração: {e}")

download_and_extract_data()

# ==============================================================================
#                               MODELO CNN (HairTypeClassifier)
# ==============================================================================

class HairTypeClassifier(nn.Module):
    """
    Define a arquitetura da CNN conforme as especificações do exercício.
    Input Shape esperado: (3, 200, 200)
    """
    def __init__(self):
        super().__init__()
        
        # 1. Camada Convolucional: Input (3 canais) -> Output (32 filtros)
        # Output shape: (32, 198, 198)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0
        )
        self.relu1 = nn.ReLU()
        
        # 2. Max Pooling: Reduz o mapa de características pela metade
        # Output shape: (32, 99, 99)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # O tamanho achatado (flattened) é 32 canais * 99 * 99 pixels
        FLATTENED_SIZE = 32 * 99 * 99 # 313632
        
        # 3. Camada Linear Oculta: 313632 -> 64 neurônios
        self.linear1 = nn.Linear(
            in_features=FLATTENED_SIZE,
            out_features=64
        )
        self.relu2 = nn.ReLU()
        
        # 4. Camada Linear de Saída: 64 -> 1 neurônio
        self.output = nn.Linear(
            in_features=64,
            out_features=1
        )
        # 5. Ativação final: Sigmoid para classificação binária
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Passo 1: Convolução -> ReLU -> Pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Passo 2: Flatten - Transforma o tensor (Batch, Channels, H, W) em (Batch, C*H*W)
        x = x.view(x.size(0), -1) 
        
        # Passo 3: Camada Oculta -> ReLU
        x = self.linear1(x)
        x = x = self.relu2(x)
        
        # Passo 4: Camada de Saída -> Sigmoid
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x

# ==============================================================================
#                           FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ==============================================================================

def train_and_validate(model, criterion, optimizer, train_loader, val_loader, num_epochs, start_epoch, history, train_dataset, validation_dataset, use_sigmoid_in_model):
    """
    Função de treinamento e validação modular.
    Adapta-se ao uso de Sigmoid no modelo vs. BCEWithLogitsLoss.
    """
    print(f"\n--- Iniciando Treinamento (Épocas {start_epoch+1} a {start_epoch+num_epochs}) ---")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # --------------------- FASE DE TREINAMENTO ---------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Unsqueeze(1) muda a shape de (batch_size) para (batch_size, 1)
            # Float() é necessário para a função de perda (BCELoss ou BCEWithLogitsLoss)
            labels = labels.float().unsqueeze(1) 

            optimizer.zero_grad() # Zera os gradientes
            
            outputs = model(images) # Forward pass
            
            # Ajusta a saída se BCEWithLogitsLoss for usada (remove sigmoid se presente)
            if use_sigmoid_in_model and isinstance(criterion, nn.BCEWithLogitsLoss):
                # Se o modelo tem Sigmoid E o criterion é BCEWithLogitsLoss, isso é um erro.
                # Para fins de execução do exercício, vamos assumir que o criterion é BCELoss.
                # Se fosse BCEWithLogitsLoss, o Sigmoid deveria ser removido do modelo.
                pass # Usamos BCELoss na prática.

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            # Cálculo de acurácia: aplica threshold de 0.5 na saída (que já tem Sigmoid)
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

        # --------------------- FASE DE VALIDAÇÃO ---------------------
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Desativa o cálculo de gradientes
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(validation_dataset)
        val_epoch_acc = correct_val / total_val
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f"Epoch {epoch+1:2d}/{start_epoch + num_epochs}, "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

# ==============================================================================
#                               EXECUÇÃO: TREINAMENTO INICIAL (QUESTÕES 3 & 4)
# ==============================================================================

# Hiperparâmetros
BATCH_SIZE = 20
NUM_EPOCHS = 10
LR = 0.002
MOMENTUM = 0.8

# 1. Definição das Transformações Iniciais (Sem Augmentation)
train_transforms_initial = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    # Normalização ImageNet (Padrão para modelos pré-treinados, aqui usado por convenção)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# As transformações de teste/validação não incluem augmentations
test_transforms = train_transforms_initial 

# 2. Carregamento dos Datasets (Inicial)
train_dataset = datasets.ImageFolder(root='./train', transform=train_transforms_initial)
test_dataset = datasets.ImageFolder(root='./test', transform=test_transforms)

# 3. DataLoaders (Shuffle=True para treino, False para teste)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Instanciação do Modelo, Otimizador e Função de Perda
model = HairTypeClassifier().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# Usamos nn.BCELoss() porque o modelo já aplica Sigmoid na saída
criterion = nn.BCELoss() 
# Se tivéssemos removido o Sigmoid do modelo, usaríamos nn.BCEWithLogitsLoss()

history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

# 5. Executa Treinamento Inicial (Epochs 1-10)
train_and_validate(
    model, criterion, optimizer, train_loader, test_loader, 
    num_epochs=NUM_EPOCHS, start_epoch=0, history=history, 
    train_dataset=train_dataset, validation_dataset=test_dataset, 
    use_sigmoid_in_model=True
)

# ==============================================================================
#                 EXECUÇÃO: TREINAMENTO COM AUGMENTATION (QUESTÕES 5 & 6)
# ==============================================================================

print("\n" + "="*80)
print("INICIANDO FASE 2: TREINAMENTO COM DATA AUGMENTATION (Epochs 11-20)")
print("="*80)

# 1. Definição das Transformações com Data Augmentation
train_transforms_aug = transforms.Compose([
    transforms.Resize((200, 200)),
    # Augmentations adicionadas
    transforms.RandomRotation(50), 
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    # Conversão e Normalização (devem ser as últimas)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. Re-carregamento do Dataset de Treino com Novas Transformações
train_dataset_aug = datasets.ImageFolder(root='./train', transform=train_transforms_aug)

# 3. Novo DataLoader de Treino
train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True)

# 4. Continua o treinamento do *mesmo modelo* por mais 10 épocas
train_and_validate(
    model, criterion, optimizer, train_loader_aug, test_loader, 
    num_epochs=NUM_EPOCHS, start_epoch=NUM_EPOCHS, history=history, 
    train_dataset=train_dataset_aug, validation_dataset=test_dataset, 
    use_sigmoid_in_model=True
)

# ==============================================================================
#                            RESUMO DOS RESULTADOS
# ==============================================================================

# Coleta os resultados do histórico para responder às questões
train_acc_q3 = history['acc'][:NUM_EPOCHS] # Acurácia das Épocas 1-10
train_loss_q4 = history['loss'][:NUM_EPOCHS] # Perda das Épocas 1-10
val_loss_q5 = history['val_loss'][NUM_EPOCHS:] # Perda de Validação das Épocas 11-20
val_acc_q6 = history['val_acc'][-5:] # Acurácia de Validação das últimas 5 épocas (16-20)

print("\n" + "#"*80)
print("# RESUMO DAS MÉTRICAS PARA AS QUESTÕES:")
print("#"*80)

# Resposta Questão 3: Mediana da Acurácia de Treino (Epochs 1-10)
median_acc_q3 = np.median(train_acc_q3)
print(f"Q3: Mediana da Acurácia de Treino (Epochs 1-10): {median_acc_q3:.4f}")

# Resposta Questão 4: Desvio Padrão da Perda de Treino (Epochs 1-10)
std_loss_q4 = np.std(train_loss_q4)
print(f"Q4: Desvio Padrão da Perda de Treino (Epochs 1-10): {std_loss_q4:.4f}")

# Resposta Questão 5: Média da Perda de Teste (Epochs 11-20)
mean_val_loss_q5 = np.mean(val_loss_q5)
print(f"Q5: Média da Perda de Teste (Validação) com Augmentation (Epochs 11-20): {mean_val_loss_q5:.4f}")

# Resposta Questão 6: Média da Acurácia de Teste (Epochs 16-20)
mean_val_acc_q6 = np.mean(val_acc_q6)
print(f"Q6: Média da Acurácia de Teste (Validação) (Últimas 5 Augmentation): {mean_val_acc_q6:.4f}")
print("#"*80)
