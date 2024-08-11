from crewai import Agent
from tools import tool

from dotenv import load_dotenv
load_dotenv()

import os

from langchain_google_genai import ChatGoogleGenerativeAI

# call gemini model

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key = os.getenv('GOOGLE_API_KEY'))


# CREATING a senior research agent with memory 

news_researcher = Agent(
    role = "Senior Researcher",
    goal =  "Uncover ground breaking technologies in {topic}",
    verbose = True,
    memory = True,
    backstory = (
        "Driven by curiosity , you are at the forefront of innovation, eager to explore and share knowledge"
    ),
    tools = [tool],
    llm=llm,
    allow_delegation = True,

)

# writer agenty with custom tools responsible for writing news blocks
news_writer = Agent(
    role = "Writer",
    goal =  "Narrate compelling tech stories about {topic}",
    verbose = True,
    memory = True,
    backstory = (
        "With a flair fro simplifying complex topics , you craft engaging narratives that captivates and educates, bringing new discoveries to light"
    ),
    tools = [tool],
    llm=llm,
    allow_delegation = False,
)
