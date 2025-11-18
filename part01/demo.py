from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather],
    system_prompt="you are a helpful assistant"
)

resp = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
print(resp)
