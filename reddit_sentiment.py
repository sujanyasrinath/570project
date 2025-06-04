import os
import boto3
import praw
import pandas as pd
from langchain import hub
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.chat_models import BedrockChat
from langchain.tools import Tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

# AWS Bedrock client setup
aws_client = boto3.client(service_name="bedrock-runtime")
llm = BedrockChat(
    client=aws_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 0.9,
        "stop_sequences": ["\n\nHuman"]
    },
)

# Prompt setup for structured chat agent
prompt = hub.pull("hwchase17/structured-chat-agent")

# Reddit & sentiment analyzer setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_for_restaurant(restaurant, limit=25):
    try:
        posts = reddit.subreddit("all").search(restaurant, limit=limit)
        titles = [post.title for post in posts if post.title]
        scores = [analyzer.polarity_scores(title)['compound'] for title in titles]
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print(f"Error processing {restaurant}: {e}")
        return 0

def get_reddit_post_titles_and_links(query, limit=10):
    try:
        posts = reddit.subreddit("all").search(query, limit=limit)
        return [{"title": post.title, "url": f"https://www.reddit.com{post.permalink}"} for post in posts]
    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        return []

def summarize_restaurant_reddit(restaurant_name: str):
    reddit_tool = Tool.from_function(
        func=lambda query: RedditSearchRun().invoke({
            "query": query,
            "limit": "10",
            "subreddit": "all",
            "time_filter": "month",
            "sort": "relevance"
        }),
        name="RedditSearch",
        description="Searches Reddit for discussions about a restaurant."
    )

    agent = create_structured_chat_agent(llm, [reddit_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[reddit_tool], verbose=True)

    query = f"""
Based on recent Reddit posts, list 3–5 concise bullet points that highlight what customers should know before visiting {restaurant_name}.

✅ Focus only on:
- Food quality, taste, and presentation
- Service and staff behavior
- Ambience and cleanliness
- Pricing and value

❌ Do NOT include:
- Basic details (location, hours, menus)
- Reddit usernames or unrelated drama

Write each point as a short, helpful sentence (max 15 words).
"""

    response = agent_executor.invoke({"input": query})
    summary = response.get("output", "") if isinstance(response, dict) else str(response)
    sentiment_score = get_sentiment_for_restaurant(restaurant_name)

    return {
        "restaurant": restaurant_name,
        "summary": summary,
        "avg_sentiment_score": sentiment_score
    }

def label_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"
