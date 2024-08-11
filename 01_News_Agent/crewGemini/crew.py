from crewai import Crew, Process
from tasks import research_task, write_task
from agents import news_researcher, news_writer

# forming tech focused crew with some enhanced confi
crew = Crew(
    agents = [news_researcher, news_writer],
    tasks = [research_task,write_task ],
    process = Process.sequential  #Optional , sequential is default

)

# start exexcution of task

result = crew.kickoff(inputs = {'topic': 'AI in Sports'})
print(result)