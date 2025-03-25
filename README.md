# Sistema Preditivo de Risco de Sinistros com Explicabilidade para Pequenas Seguradoras

## Descrição Técnica  
Este projeto implementa um sistema de machine learning para prever a probabilidade de clientes de uma seguradora sofrerem sinistros no próximo ano, combinando técnicas avançadas de modelagem preditiva com explicações interpretáveis dos fatores de risco. O sistema utiliza dados históricos dos clientes (demográficos, comportamentais e veiculares) para:  

1. **Previsão de Risco Temporal:**  
   - Cria uma variável alvo (`proximo_sinistro`) através de deslocamento temporal, garantindo que o modelo aprenda padrões sequenciais  
   - Mantém a integridade temporal na divisão treino/teste para evitar vazamento de dados  

2. **Pipeline de Processamento:**  
   - Padronização de features numéricas (idade, score de crédito, histórico de sinistros)  
   - Codificação de variáveis categóricas (sexo)  
   - Modelagem com XGBoost otimizado para dados desbalanceados via `scale_pos_weight`  

3. **Explicabilidade com SHAP:**  
   - Gera explicações individuais para cada previsão  
   - Identifica os 3 principais fatores de risco por cliente  
   - Quantifica a contribuição relativa de cada fator em porcentagem  

4. **Saída Operacional:**  
   - Classificação de risco em 4 níveis (Extremo/Alto/Moderado/Baixo)  
   - Ranking dos 20 clientes com maior risco  
   - Relatório técnico com métricas de validação e limitações do modelo  

## Diferenciais Chave  
- Combina previsão quantitativa com explicação qualitativa  
- Adaptado para dados temporais sequenciais de clientes  
- Sistema autoexplicativo para tomada de decisão regulatória  
- Detecção proativa de clientes em risco extremo (>95% probabilidade)  

## Aplicação Prática  
Permite que seguradoras priorizem ações preventivas, personalizem prêmios de seguros, e cumpram requisitos regulatórios de transparência em modelos de risco.  
