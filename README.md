# Regressão Polinomial para Módulo de Resiliência (MR)

Este aplicativo Streamlit permite:

- Fazer o upload de uma tabela contendo os parâmetros:
  - **σ₃**: Tensão confinante (MPa)
  - **σ_d**: Tensão desvio (MPa)
  - **MR**: Módulo de Resiliência (MPa)
- Escolher o grau da equação polinomial (mínimo 2, máximo 6; padrão 2)
- Selecionar o tipo de energia (Normal, Intermediária ou Modificada; padrão Normal)
- Calcular a equação de regressão polinomial, exibindo:
  - A equação em formato LaTeX (para formatação de subíndices e expoentes)
  - Indicadores estatísticos: R², R² ajustado, RMSE, MAE e o intercepto (com interpretações dos indicadores, exceto o intercepto)
  - Um gráfico 3D da superfície ajustada (σ₃ no eixo X, σ_d no eixo Y e MR no eixo Z)
- Salvar um relatório em Word com todos os dados da regressão (equação, indicadores e gráfico)

## Como executar

1. Clone este repositório:
    ```bash
    git clone <URL-do-repositório>
    cd <nome-do-repositório>
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Execute o aplicativo:
    ```bash
    streamlit run app.py
    ```

4. Siga as instruções na interface web para fazer o upload da tabela e ajustar as configurações do modelo.

## Formato da Tabela de Exemplo

A tabela deve conter as seguintes colunas (com cabeçalhos exatamente como abaixo):
- `σ3`
- `σd`
- `MR`
