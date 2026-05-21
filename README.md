# Memória do Projeto

## Contexto
Este projeto visa automatizar o ajuste de modelos de regressão para o Módulo de Resiliência (MR) em solos e materiais de pavimentação, baseando-se em ensaios triaxiais.

## Histórico de Decisões
- **2024-05-22:** Início da reestruturação profissional.
- **2024-05-22:** Decisão de modularizar o projeto em `models/` e `utils/` para facilitar a manutenção e escalabilidade.
- **2024-05-22:** Implementação dos 14 modelos selecionados na dissertação de Camila Luiza Mello Carvalho (UFJF, 2023), baseando-se no arquivo LaTeX disponibilizado pelo usuário.
- **2024-05-22:** Adição de modelos genéricos (Polinomiais e Potência Composta) para fins comparativos.
- **2024-05-22:** Implementação de suíte de testes automatizados com `pytest` para garantir integridade matemática.

## Estado Atual
- Modelos implementados: 17 modelos no total (14 de referência + 3 genéricos).
- Estrutura: Modular, orientada a objetos (BaseModel).
- Interface: Streamlit dinâmica baseada no `MODELS_MAP`.
- Testes: 100% de cobertura nos métodos de fit/predict de todos os modelos.

## Próximos Passos
1. Monitorar feedback dos usuários sobre a precisão dos ajustes.
2. Considerar a implementação de modelos de rede neural como alternativa aos modelos matemáticos tradicionais.
