#         Sistema de Recomendação por Similaridade

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

Sistema de recomendação de produtos de moda baseado em similaridade visual usando Transfer Learning com PyTorch.

</div>

## 🎯 Objetivo

Desenvolver um sistema inteligente que recomenda produtos de moda similares através da análise visual de imagens, utilizando redes neurais convolucionais pré-treinadas para extração de características visuais e cálculo de similaridade.

## ✨ Características

- **Transfer Learning** com ResNet50 pré-treinado
- **Extração de Features** de 2048 dimensões
- **Busca por Similaridade** usando cosine similarity
- **Interface Interativa** para teste com imagens da internet
- **Visualização Automática** dos resultados
- **Suporte a Lotes** para análise em massa

## 🛠️ Tecnologias Utilizadas

- **PyTorch** - Framework de Deep Learning
- **Torchvision** - Modelos pré-treinados e transformações
- **ResNet50** - Rede neural para extração de features
- **Scikit-learn** - Cálculo de métricas de similaridade
- **PIL/OpenCV** - Processamento de imagens
- **Matplotlib** - Visualização dos resultados
- **NumPy/Pandas** - Manipulação de dados

## 📊 Dataset

- **Fashion Product Images Small** (Kaggle)
- Contém produtos de moda diversos: roupas, calçados, acessórios
- Imagens em diferentes ângulos e backgrounds
- Categorias variadas para teste de generalização

## 🚀 Instalação e Uso

### 1. Pré-requisitos

```bash
# Clone o repositório
git clone https://github.com/JohnnyPassos/recomendacao_similaridade.git
cd recomendacao_similaridade

# Instale as dependências
pip install -r requirements.txt
```

### 2. Configuração Inicial

```python
# Importar bibliotecas principais
import torch
import torchvision.models as models
from src.feature_extractor import ImageFeatureExtractor
from src.similarity_recommender import ImageSimilarityRecommender

# Verificar dispositivo disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo utilizado: {device}')
```

### 3. Uso Básico

```python
# Inicializar o extrator de features
feature_extractor = ImageFeatureExtractor(model_name='resnet50', device=device)

# Criar sistema de recomendação
recommender = ImageSimilarityRecommender(feature_extractor)

# Construir índice do dataset
recommender.build_index('path/to/fashion/dataset')

# Buscar imagens similares
similar_images = recommender.find_similar_images(
    query_image_path='test_image.jpg', 
    top_k=5, 
    metric='cosine'
)

# Visualizar resultados
recommender.visualize_results('test_image.jpg', similar_images)
```

### 4. Teste com Imagens da Internet

```python
# Baixar e testar imagem da internet
from src.utils import download_image_from_url, test_with_internet_image

# URL de exemplo
url = "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400"
test_image_path, results = test_with_internet_image(url, recommender)
```

## 📁 Estrutura do Projeto

```
FashionSimilarityRecommendation/
├── README.md
├── requirements.txt
├── notebooks/
│   └── Fashion_Similarity_Complete.ipynb
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py      # Classe para extração de features
│   ├── similarity_recommender.py # Sistema de recomendação
│   └── utils.py                  # Funções auxiliares
├── models/
│   └── fashion_similarity_index.pkl  # Índice pré-computado
├── examples/
│   ├── test_images/             # Imagens de exemplo
│   └── demo_usage.py            # Script de demonstração
├── results/
│   └── similarity_examples/     # Resultados de exemplo
└── docs/
    └── methodology.md           # Documentação técnica
```

## 🔬 Metodologia

### Extração de Features
1. **Pré-processamento**: Redimensionamento para 224x224, normalização ImageNet
2. **Backbone**: ResNet50 pré-treinado (sem camada de classificação final)
3. **Features**: Vetor de 2048 dimensões por imagem
4. **Otimização**: Processamento em lotes para eficiência

### Cálculo de Similaridade
- **Métrica Principal**: Cosine Similarity
- **Alternativa**: Distância Euclidiana
- **Ranking**: Ordenação decrescente por similaridade
- **Threshold**: Configurável para filtrar resultados

## 📈 Resultados

### Performance
- **Tempo de Indexação**: ~2-3 segundos por 100 imagens
- **Tempo de Busca**: <1 segundo para dataset de 1000+ imagens
- **Precisão Visual**: Alta correlação com similaridade percebida

### Exemplos de Uso
- ✅ Recomendação de roupas similares
- ✅ Busca por produtos complementares
- ✅ Análise de tendências de moda
- ✅ Organização automática de catálogos

## 🧪 Testes

### Executar Testes Básicos
```bash
# Teste com dataset local
python examples/demo_usage.py --dataset_path ./fashion_dataset

# Teste interativo
python examples/interactive_test.py

# Teste em lote
python examples/batch_test.py
```

### Métricas de Avaliação
- **Top-K Accuracy**: Precisão nos K primeiros resultados
- **Mean Average Precision (MAP)**: Qualidade geral do ranking
- **Visual Similarity Score**: Avaliação qualitativa

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📚 Outros Projetos

Confira meus outros repositórios de Computer Vision e Machine Learning:

- 🎯 [**DeteccaoRedeYolov8**](https://github.com/JohnnyPassos/DeteccaoRedeYolov8) - Detecção de objetos com YOLOv8
- 👤 [**DeteccaoeClassificacaodeFace**](https://github.com/JohnnyPassos/DeteccaoeClassificacaodeFace) - Detecção e classificação facial
- 📊 [**calculo_metricas_ML**](https://github.com/JohnnyPassos/calculo_metricas_ML) - Cálculo de métricas em Machine Learning
- 🚀 [**desafio2-bootcamp-ML-DIO**](https://github.com/JohnnyPassos/desafio2-bootcamp-ML-DIO) - Projetos do bootcamp ML

## 👨‍💻 Autor

**Johnny Passos**
- GitHub: [@JohnnyPassos](https://github.com/JohnnyPassos)
- Especialista em Computer Vision e Machine Learning
- Experiência em PyTorch, TensorFlow, OpenCV e YOLO

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.

## 🙏 Agradecimentos

- Equipe do PyTorch pela excelente documentação
- Comunidade Kaggle pelo dataset Fashion Product Images
- Contribuidores do torchvision pelos modelos pré-treinados

---

<div align="center">

⭐ **Se este projeto foi útil, considere dar uma estrela!** ⭐

</div>
