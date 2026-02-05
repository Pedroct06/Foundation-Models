# PatchTST - Previsão de Séries Temporais

Este notebook implementa o modelo **PatchTST** (Patch Time Series Transformer) para previsão de séries temporais usando a biblioteca NeuralForecast.

## 📋 Descrição

O projeto demonstra a aplicação do PatchTST em diferentes conjuntos de dados de séries temporais, incluindo:

- 🚗 **Viagens de veículos** (vehicle trips)
- 🔧 **Vendas de peças automotivas** (car parts)
- ₿ **Preço do Bitcoin**
- 🏧 **Saques em caixas eletrônicos** (NN5 daily)
- ☀️ **Manchas solares** (sunspot) - ciclo solar de 11 anos

## 🛠️ Requisitos

```bash
pip install neuralforecast
```

Bibliotecas utilizadas:
- pandas
- numpy
- neuralforecast
- matplotlib
- pytorch-lightning

## 📁 Estrutura do Código

### Funções Principais

#### `parse_tsf(file_path, frequency='D')`
Parseia arquivos no formato TSF (Time Series Format) e retorna um DataFrame com colunas:
- `unique_id`: identificador da série
- `ds`: timestamp
- `y`: valor observado

#### `filtrar_serie(df, id_serie=None)`
Filtra uma série específica do dataset. Se nenhum ID for fornecido, retorna a primeira série disponível.

#### `run_patchtst(df, horizon, frequency='D')`
Executa o modelo PatchTST com os seguintes parâmetros:
- **horizon**: número de períodos a prever
- **input_size**: 3× o horizonte de previsão
- **max_steps**: 300 épocas de treinamento
- **scaler_type**: normalização minmax

Divide os dados em treino/teste e retorna as previsões.

#### `calculate_metrics(y_test, y_hat)`
Calcula métricas de erro:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **SMAPE** (Symmetric Mean Absolute Percentage Error)

#### `plot_forecast(results)`
Visualiza a comparação entre valores reais e previstos.

## 🚀 Como Usar

### Exemplo Básico

```python
# Carregar dados
df = parse_tsf('seu_arquivo.tsf', frequency='D')

# Filtrar uma série específica
df_filtrado = filtrar_serie(df, id_serie="T37")

# Fazer previsão para os próximos 5 períodos
y_test, y_hat = run_patchtst(df_filtrado, horizon=5)

# Calcular métricas
mae_val, rmse_val, mape_val, smape_val = calculate_metrics(y_test, y_hat)

# Visualizar resultados
plot_forecast(y_test.merge(y_hat, on=['unique_id', 'ds'], how='left'))
```

## 📊 Exemplos de Aplicação

### 1. Viagens de Veículos
```python
df = parse_tsf('vehicle_trips_dataset_without_missing_values.tsf', frequency='D')
df_filtrado = filtrar_serie(df, id_serie="T37")
y_test, y_hat = run_patchtst(df_filtrado, 5)
```

### 2. Vendas de Peças Automotivas
```python
df = parse_tsf('car_parts_dataset_without_missing_values.tsf')
df_filtrado = filtrar_serie(df, id_serie="T1032")
y_test, y_hat = run_patchtst(df_filtrado, 5)
```

### 3. Preço do Bitcoin
```python
df = parse_tsf('bitcoin_dataset_without_missing_values.tsf')
df_filtrado = filtrar_serie(df, id_serie='price')
y_test, y_hat = run_patchtst(df_filtrado, 3)
```

### 4. Saques em Caixas Eletrônicos
```python
df = parse_tsf('nn5_daily_dataset_without_missing_values.tsf', frequency='D')
df_filtrado = filtrar_serie(df, id_serie="T27")
y_test, y_hat = run_patchtst(df_filtrado, 3)
```

### 5. Manchas Solares
```python
df = parse_tsf('sunspot_dataset_without_missing_values.tsf')
df_filtrado = filtrar_serie(df)
y_test, y_hat = run_patchtst(df_filtrado, 5)
```

> **Nota:** As manchas solares apresentam um ciclo de aproximadamente 11 anos, o que explica variações abruptas nos dados.

## 📈 Sobre o Modelo PatchTST

O PatchTST é um modelo baseado em Transformers que:
- Divide séries temporais em patches (segmentos)
- Utiliza atenção multi-cabeça para capturar dependências temporais
- Oferece boa performance com eficiência computacional
- É adequado para séries temporais univariadas e multivariadas

## 🎯 Métricas de Avaliação

- **MAE**: Erro médio absoluto - quanto menor, melhor
- **RMSE**: Raiz do erro quadrático médio - penaliza erros grandes
- **MAPE**: Erro percentual médio absoluto - interpretável como porcentagem
- **SMAPE**: Versão simétrica do MAPE - varia de 0 a 200%

## 📝 Observações

- O código suprime warnings e logs do PyTorch Lightning para limpeza visual
- Os dados de teste correspondem aos últimos `horizon` pontos da série
- O modelo usa normalização minmax para estabilizar o treinamento
- A visualização permite comparação direta entre valores reais e previstos

## 🔗 Links Úteis

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [PatchTST Paper](https://arxiv.org/abs/2211.14730)

## 📄 Licença

Este notebook está disponível no GitHub: [Foundation-Models](https://github.com/Pedroct06/Foundation-Models)
