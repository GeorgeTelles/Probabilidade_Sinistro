import pandas as pd
import numpy as np

np.random.seed(42)

dados = []

# Definir ano base para simulação
ano_base = np.random.randint(2010, 2023)

for cliente_id in range(1, 3001):
    # Gerar características base fixas
    sexo = np.random.choice(['M', 'F'])
    idade_inicial = np.random.randint(18, 60)
    veiculo_idade_inicial = np.random.randint(0, 5)
    tempo_cliente_total = np.random.randint(5, 11)
    ano_inicio = ano_base - tempo_cliente_total 
    
    historico_sinistros = 0
    score_credito = np.clip(np.random.normal(700, 50), 300, 850)
    infracoes_acumulado = 0
    
    # Gerar histórico anual
    for ano_relativo in range(1, tempo_cliente_total + 1):
        ano_atual = ano_inicio + ano_relativo
        
        # Atualizar variáveis temporais
        idade = idade_inicial + ano_relativo - 1
        veiculo_idade = veiculo_idade_inicial + ano_relativo - 1
        
        # Gerar variáveis anuais
        quilometragem_anual = np.random.randint(5000, 35000) + np.random.randint(-1000, 1000)
        infracoes_ano = np.random.poisson(0.3 + infracoes_acumulado*0.1)
        infracoes_acumulado += infracoes_ano
        
        # Atualizar score de crédito
        score_credito = np.clip(
            score_credito + np.random.normal(2, 5) - (infracoes_ano*5) - (historico_sinistros*10),
            300, 850
        )
        
        # Calcular risco
        risco = 0.1 + \
               (veiculo_idade * 0.03) + \
               (quilometragem_anual/50000) + \
               ((800 - score_credito)/500) + \
               (infracoes_acumulado * 0.2) + \
               (historico_sinistros * 0.4)
        
        prob_sinistro = 1 / (1 + np.exp(-risco))
        sinistro = np.random.binomial(1, prob_sinistro)
        
        # Adicionar registro
        dados.append([
            cliente_id,
            idade,
            sexo,
            ano_relativo,
            ano_atual,    
            veiculo_idade,
            quilometragem_anual,
            round(score_credito, 2),
            infracoes_ano,
            historico_sinistros,
            sinistro
        ])
        
        historico_sinistros += sinistro

colunas = [
    'cliente_id',
    'idade',
    'sexo',
    'tempo_cliente_anos',
    'ano',
    'idade_veiculo',
    'quilometragem_anual',
    'score_credito',
    'infracoes_transito_ano',
    'historico_sinistros',
    'sinistro'
]

df = pd.DataFrame(dados, columns=colunas)
df = df.sort_values(['cliente_id', 'ano'])

# Salvar em Excel
df.to_excel('dados_historicos_clientes.xlsx', index=False)
print("Arquivo gerado com sucesso!")