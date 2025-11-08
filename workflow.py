# workflow.py
from crewai import Crew
from agents import schema_agent, imbalance_agent, preprocess_agent

crew = Crew(agents=[schema_agent, imbalance_agent, preprocess_agent])
crew_result = crew.kickoff(inputs={"dataset_path": "data/input.csv"})
print(crew_result)