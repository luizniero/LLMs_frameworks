# pip install crewai
import os
os.environ["PYTHONUTF8"] = "1"

import warnings
warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew
from crewai.tools import SerperDevTool

# ================================================
# FERRAMENTAS (Opcional: vamos dar uma ferramenta de busca para o pesquisador)
# ================================================

search_tool = SerperDevTool()  # usa uma ferramenta de busca no Google

# ================================================
# DEFINIÇÃO DOS AGENTES
# ================================================

# Agente 1 - Pesquisador
pesquisador = Agent(
    role='Pesquisador de Tecnologia',
    goal='Buscar informações confiáveis e atualizadas sobre qualquer tema de tecnologia.',
    backstory=(
        "Você é um pesquisador dedicado. Seu trabalho é encontrar informações atualizadas e confiáveis."
    ),
    verbose=True,
    tools=[search_tool],  # o agente pode usar ferramentas externas, como buscadores
)

# Agente 2 - Redator
redator = Agent(
    role='Redator de Tecnologia',
    goal='Escrever textos claros e envolventes baseados nas informações que receber.',
    backstory=(
        "Você é um excelente redator técnico que consegue transformar informações em textos claros e atraentes."
    ),
    verbose=True
)

# ================================================
# DEFINIÇÃO DA TAREFA
# ================================================

# Task do Pesquisador
task_pesquisar = Task(
    description=(
        "Pesquise e reúna informações confiáveis sobre o tema: 'tendências em Inteligência Artificial para 2025'."
    ),
    expected_output="Lista resumida com pelo menos 5 tendências importantes em IA para 2025."
)

# Task do Redator
task_redigir = Task(
    description=(
        "Usando as informações obtidas pelo pesquisador, redija um texto objetivo e interessante sobre as tendências em IA para 2025."
    ),
    expected_output="Um artigo curto de até 300 palavras, em linguagem clara e envolvente."
)

# ================================================
# CRIAÇÃO DA CREW (a equipe)
# ================================================

crew = Crew(
    agents=[pesquisador, redator],
    tasks=[task_pesquisar, task_redigir],
    verbose=True
)

# ================================================
# EXECUTANDO O PROJETO
# ================================================

result = crew.kickoff()
print("\n========== Resultado Final ==========\n")
print(result)
