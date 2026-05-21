# Modelos de Regressão para Módulo de Resiliência (MR)

Este aplicativo Streamlit realiza o ajuste de diversos modelos de regressão ao Módulo de Resiliência (MR) a partir de dados de ensaios triaxiais.

## Modelos Disponíveis
O projeto conta com 17 modelos de regressão, incluindo os 14 modelos selecionados na dissertação de **Camila Luiza Mello Carvalho (UFJF, 2023)**:
- Dunlap (1963)
- Hicks (1970)
- Witczak (1981)
- Uzan (1985)
- Johnson et al. (1986)
- Witczak e Uzan (1988)
- Tam e Brown (1988)
- Pezo (1993)
- Hopkins et al. (2001)
- Ni et al. (2002)
- NCHRP 1-28A (2004)
- NCHRP 1-37A (2004)
- Ooi et al. (1) (2004)
- Ooi et al. (2) (2004)
- Modelos Polinomiais (2º ao 6º grau)
- Modelo de Potência Composta Genérico

## Funcionalidades
- **Ajuste de Modelos:** Cálculo automático de coeficientes através de regressão linear e não-linear (`curve_fit`).
- **Indicadores Estatísticos:** R², R² Ajustado, RMSE, MAE, NRMSE e CV(RMSE).
- **Visualização 3D:** Gráficos interativos da superfície de resposta (MR vs σ₃ vs σd).
- **Exportação:** Geração de relatórios profissionais em Word (.docx) e pacotes LaTeX (.zip).

## Instalação e Execução Local

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o app:
   ```bash
   streamlit run app.py
   ```

## Testes
Para garantir a integridade dos modelos, execute a suíte de testes:
```bash
python -m pytest tests/test_models.py
```

## Referência Bibliográfica
CARVALHO, Camila Luiza Mello. **Avaliação de modelos matemáticos de comportamento para o Módulo de Resiliência de Materiais de Pavimentação**. 2023. Dissertação (Mestrado em Engenharia Civil) - Universidade Federal de Juiz de Fora, Juiz de Fora, 2023.
