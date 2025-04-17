# Modelos de Regressão para Módulo de Resiliência (MR)

Este repositório contém um aplicativo **Streamlit** para ajuste de modelos de regressão ao Módulo de Resiliência (MR) em ensaios triaxiais.

## Funcionalidades

- **Upload** de arquivo CSV ou XLSX com colunas exatas:
  - `σ3` (σ₃) – tensão confinante [MPa]
  - `σd` (σₖₗ) – tensão desvio [MPa]
  - `MR` – módulo de resiliência [MPa]

- **Escolha do modelo de regressão**:
  - Polinomial c/ Intercepto
  - Polinomial s/Intercepto
  - Potência Composta c/Intercepto
  - Potência Composta s/Intercepto
  - Pezo (não normalizado)
  - Pezo (original) – usa Pa = 0,101325 MPa

- **Configurações adicionais**:
  - Grau polinomial de 2 a 6 (apenas para modelos polinomiais)
  - Tipo de energia: Normal (padrão), Intermediária ou Modificada

- **Resultados exibidos**:
  - **Equação ajustada** em LaTeX (com subíndices e expoentes corretamente formatados)
  - **Indicadores estatísticos**:
    - R²
    - R² ajustado (sempre ≤ R² e ≤ 1)
    - RMSE [MPa]
    - MAE [MPa]
    - Média e desvio padrão de MR
    - Intercepto
  - **Gráfico 3D interativo** da superfície de MR (Plotly)
  - **Download** de relatório em Word (.docx) contendo:
    - Configurações usadas
    - Equação no formato nativo do Word (via função Equação)
    - Tabela de dados de entrada
    - Gráfico 3D

## Formato do arquivo de dados

O arquivo deve conter cabeçalho exato: `σ3`, `σd`, `MR`. Use ponto (`.`) como separador decimal, ou CSV com vírgula (`decimal=","` no app).

Exemplo de planilha:

| σ3    | σd    | MR    |
|------:|------:|------:|
| 0.020 | 0.020 | 255.0 |
| 0.020 | 0.040 | 191.0 |
| ...   | ...   | ...   |

## Instalação e execução local

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   cd SEU_REPOSITORIO
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o app:
   ```bash
   streamlit run app.py
   ```

## Deploy no Streamlit Cloud

1. Faça push do repositório para o GitHub.
2. No **Streamlit Cloud**, selecione **New app** → vincule seu repositório.
3. Defina o comando de execução: `streamlit run app.py`.
4. Habilite rede auto-deploy (opcional).

## Dependências principais

- Python 3.7+
- streamlit
- pandas
- numpy
- scikit-learn
- scipy
- plotly
- python-docx

## Licença

Este projeto está licenciado sob a licença [MIT](LICENSE).
