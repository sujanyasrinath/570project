from langchain_community.chat_models import BedrockChat
from langchain.agents import initialize_agent, Tool
from dish_tool import analyze_top_dishes

def run_dish_agent():
    llm = BedrockChat(
        model_id="anthropic.claude-v2",  # or claude-v3, etc.
        region_name="us-east-1",
        model_kwargs={"temperature": 0.3}
    )

    tools = [
        Tool(
            name="AnalyzeTopDishes",
            func=analyze_top_dishes,
            description="Extracts top-rated dishes for nearby similar businesses."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-zero-shot-react-description",
        verbose=True
    )

    query = "What dishes are working well at Indian restaurants near San Luis Obispo?"
    response = agent.run(query)
    print(response)

if __name__ == "__main__":
    run_dish_agent()
