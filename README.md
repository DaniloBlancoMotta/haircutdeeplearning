# Hair Type Classifier

> **Status:** Operational 🟢  
> **Tech Stack:** PyTorch | Computer Vision | CNN

Um classificador de imagens robusto e direto ao ponto para distinguir entre cabelos **Lisos** e **Cacheados/Crespos**. Construído com PyTorch, focado em reprodutibilidade e eficiência.

##  Arquitetura Industrial (The Engine)

O "motor" desse projeto é uma CNN (Convolutional Neural Network) customizada, projetada para processamento rápido e eficaz.

*   **Input Pipeline:** Imagens RGB redimensionadas para `200x200`.
*   **Feature Extraction:**
    *   `Conv2d`: 32 filtros, kernel 3x3 (Captura de texturas e bordas).
    *   `ReLU`: Ativação não-linear.
    *   `MaxPool2d`: Downsampling 2x2 (Redução de dimensionalidade).
*   **Classification Head:**
    *   `Flatten`: Vetorização.
    *   `Linear (Dense)`: 64 neurônios ocultos.
    *   `Output`: Sigmoid (Probabilidade binária: 0=Liso, 1=Cacheado).

## 🛠️ Setup & Run

Sem enrolação. Para rodar essa máquina na sua estação de trabalho:

### 1. Prepare o Ambiente
Certifique-se de ter Python instalado. Instale as dependências blindadas:

```bash
pip install -r requirements.txt
```

### 2. Execute o Pipeline
O script cuida de tudo: baixa o dataset, extrai, treina e valida.

```bash
python hair_type_classifier.py
```

## 📊 O que esperar (Metrics)

O sistema executa em duas fases críticas:

1.  **Warm-up (Epochs 1-10):** Treinamento padrão para estabilização dos pesos.
2.  **Hardening (Epochs 11-20):** Introdução de **Data Augmentation** (rotação, crop, flip) para garantir generalização e robustez contra variações do mundo real.

Ao final, o console cuspirá as métricas de performance (Loss & Accuracy) para auditoria.

---
*Built for efficiency.* 
