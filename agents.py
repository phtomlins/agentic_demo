# https://medium.com/@whyamit101/local-ai-using-ollama-with-agents-114c72182c97

from crewai import Crew
from crewai.agents import Agent
from crewai.llms import OllamaLLM

llm = OllamaLLM(model="mistral")

# agents.py
from crewai.agents import Agent

schema_agent = Agent(
    role="Schema Validator",
    goal="Identify schema issues in tabular data",
    backstory="An expert in data quality assurance",
    llm=llm
)

imbalance_agent = Agent(
    role="Class Imbalance Detector",
    goal="Detect and summarize class imbalance issues",
    backstory="A statistical detective for unbalanced datasets",
    llm=llm
)

preprocess_agent = Agent(
    role="Preprocessing Advisor",
    goal="Recommend preprocessing steps based on input data characteristics",
    backstory="A data engineer who loves clean inputs",
    llm=llm
)