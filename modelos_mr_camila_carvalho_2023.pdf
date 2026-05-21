# Orientações para Agentes e Desenvolvedores

Este documento contém diretrizes para o desenvolvimento e manutenção do projeto de Modelagem de Módulo de Resiliência (MR).

## Arquitetura do Projeto
O projeto segue uma estrutura modular para separar a lógica de negócio (modelos matemáticos) da interface do usuário (Streamlit).

- `models/`: Contém a definição dos modelos de regressão.
  - `base_model.py`: Define a interface `BaseModel` que todos os modelos devem seguir.
  - `camila_batchX.py`: Modelos específicos da dissertação de Camila Carvalho (2023).
  - `polynomial.py` / `composite_power.py`: Modelos genéricos/polinomiais.
- `utils/`: Funções de suporte.
  - `metrics.py`: Cálculo de R², RMSE, MAE e indicadores de qualidade.
  - `plotting.py`: Geração de superfícies 3D interativas com Plotly.
  - `reports.py`: Geração de relatórios em Word (.docx) e LaTeX (.zip).
- `tests/`: Suíte de testes automatizados.
- `app.py`: Ponto de entrada da aplicação Streamlit.

## Como adicionar um novo modelo
1. Crie uma nova classe em `models/` que herde de `BaseModel`.
2. Implemente os métodos `fit(X, y)`, `predict(X)`, `get_equation()` e a propriedade `name`.
3. Adicione o novo modelo ao `MODELS_MAP` em `models/__init__.py`.
4. Execute os testes com `python -m pytest tests/test_models.py` para garantir que o modelo está funcionando corretamente.

## Princípios de Design
1. **Modularidade:** A lógica de cálculo deve ser isolada.
2. **Precisão Matemática:** Seguir rigorosamente as fórmulas da literatura.
3. **Padrões de LaTeX:** Manter as equações formatadas em LaTeX para exibição e exportação.
