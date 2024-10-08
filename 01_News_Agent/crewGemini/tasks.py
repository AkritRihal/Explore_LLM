from crewai import Task
from tools import tool
from agents import news_researcher, news_writer

# research task
research_task = Task(
    description = (
        "Identify the next big trend in {topic}. Focus on identifying pros and cons and the overall narative"
        "Your final report should clearly articulate the points,its market oppurtinities and potentisl risks "
    ),
    expected_output = "A comprehensive 3 paragraph long report on the latest AI trends",
    tools = [tool],
    agent = news_researcher
)

# writing task with language model confi

write_task = Task(
    description = (
        "Compose an insightful article on {topic}."
        "Focus on the latest trends and how it is impacting the industry. This article should be easy to understand , engaging , and positive."
    ),
    expected_output = "A comprehensive 4 paragraph article on {topic} advancements formatted as markdown",
    tools = [tool],
    agent = news_writer,
    async_execution = False,
    output_file = "new-blog-post.md"
)