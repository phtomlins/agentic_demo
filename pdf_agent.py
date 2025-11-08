from langchain_community.utilities import GoogleSerperAPIWrapper
from crewai.tools import BaseTool
from pydantic import Field
from crewai.tools import PDFSearchTool


from crewai import Crew
from crewai.agents import Agent, Task
from crewai.llms import OllamaLLM

llm = OllamaLLM(model="gemma3")

# agents.py
from crewai.agents import Agent


import requests
#paper url 
pdf_url='https://arxiv.org/pdf/2405.01577' 
response = requests.get(pdf_url)

with open('hatespeech.pdf', 'wb') as file:
    file.write(response.content)


Router_Agent = Agent(
  role='Router',
  goal='Route user question to a vectorstore or web search',
  backstory=(
    "You are an expert at routing a user question to a vectorstore or web search ."
    "Use the vectorstore for questions on hate speech or tiny llm or finetuning of llm."
    "use web-search for question on latest news or recent topics."
    "use generation for generic questions otherwise"
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)
Retriever_Agent = Agent(
role="Retriever",
goal="Use the information retrieved from the vectorstore to answer the question",
backstory=(
    "You are an assistant for question-answering tasks."
    "Use the information present in the retrieved context to answer the question."
    "You have to provide a clear concise answer within 200 words."
),
verbose=True,
allow_delegation=False,
llm=llm,
)


search = GoogleSerperAPIWrapper

class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
    search: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"
class GenerationTool(BaseTool):
    name: str = "Generation_tool"
    description: str = "Useful for generic-based queries. Use this to find  information based on your own knowledge."
    #llm: ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def _run(self, query: str) -> str:
      llm=OllamaLLM(model_name="gemma3", temperature=0)
      """Execute the search query and return results"""
      return llm.invoke(query)
generation_tool=GenerationTool()
web_search_tool = SearchTool()
pdf_search_tool = PDFSearchTool(
    pdf="hatespeech.pdf",
)

router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "Based on the keywords decide whether it is eligible for a vectorstore search or a web search or generation."
    "Return a single word 'vectorstore' if it is eligible for vectorstore search."
    "Return a single word 'websearch' if it is eligible for web search."
    "Return a single word 'generate' if it is eligible for generation."
    "Do not provide any other premable or explaination."
    ),
    expected_output=("Give a  choice 'websearch' or 'vectorstore' or 'generate' based on the question"
    "Do not provide any other premable or explaination."),
    agent=Router_Agent,
   )

retriever_task = Task(
    description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
    "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
    "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
    "otherwise generate the output basedob your own knowledge in case the router task output is 'generate"
    ),
    expected_output=("You should analyse the output of the 'router_task'"
    "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
    "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
    "If the response is 'generate' then use then use generation_tool ."
    "otherwise say i dont know if you dont know the answer"

    "Return a claer and consise text as response."),
    agent=Retriever_Agent,
    context=[router_task],
    tools=[pdf_search_tool,web_search_tool,generation_tool],
)

rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent], 
    tasks=[router_task, retriever_task],
    verbose=True,

)

result = rag_crew.kickoff(inputs={"question":"what is a llm finetuning"})