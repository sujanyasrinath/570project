import os
import boto3
import praw
import pandas as pd
from langchain import hub
#from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_aws import ChatBedrock
from langchain.tools import Tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

# AWS Bedrock client setup
aws_client = boto3.client(service_name="bedrock-runtime")
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-west-2",
    streaming=False,
    model_kwargs={
        "max_tokens": 768,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 0.9,
        "stop_sequences": ["\n\nHuman"]
    }
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
# Custom Reddit search function using PRAW
def search_reddit(query: str, limit=25, time_filter="month", sort="relevance"):
    try:
        food_keywords = ["food", "menu", "dish", "eat", "order", "meal", "popular", "tasty", "recommend"]
        subreddits = ["fastfood", "food", "restaurants", "AskReddit"]
        subreddit = reddit.subreddit("+".join(subreddits))

        posts = subreddit.search(query, limit=limit, sort=sort, time_filter=time_filter)

        filtered = [
            post for post in posts
            if any(
                keyword in post.title.lower() or keyword in post.selftext.lower()
                for keyword in food_keywords
            )
        ]

        if not filtered:
            return "No food-related posts found."

        return "\n".join(f"{post.title} (Score: {post.score})" for post in filtered)

    except Exception as e:
        return f"Error searching Reddit: {e}"





def get_sentiment_for_restaurant(restaurant, limit=25):
    try:
        posts = reddit.subreddit("all").search(restaurant, limit=limit)
        titles = [post.title for post in posts if post.title]
        scores = [analyzer.polarity_scores(title)['compound'] for title in titles]
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print(f"Error processing {restaurant}: {e}")
        return 0

def get_reddit_post_titles_and_links(query, limit=25, time_filter="month", sort="relevance"):
    try:
        food_keywords = ["food", "menu", "dish", "eat", "order", "meal", "popular", "tasty", "recommend"]
        subreddits = ["fastfood", "food", "restaurants", "AskReddit"]
        subreddit = reddit.subreddit("+".join(subreddits))

        posts = subreddit.search(query, limit=limit, sort=sort, time_filter=time_filter)

        filtered = [
            post for post in posts
            if any(
                keyword in post.title.lower() or keyword in post.selftext.lower()
                for keyword in food_keywords
            )
        ]

        return [{"title": post.title, "url": f"https://www.reddit.com{post.permalink}"} for post in filtered]

    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        return []


def summarize_restaurant_reddit(restaurant_name: str):
    reddit_tool = Tool.from_function(
    func=search_reddit,
    name="RedditSearch",
    description="Searches Reddit for discussions about a restaurant."
)


    agent = create_structured_chat_agent(llm, [reddit_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[reddit_tool], verbose=True)

    query = f"""
Summarize Reddit feedback on {restaurant_name} in 3–5 helpful bullets about food, service, or pricing.
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