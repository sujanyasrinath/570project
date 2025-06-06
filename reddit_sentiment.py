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
import spacy
from spacy.matcher import PhraseMatcher

load_dotenv()

# AWS Bedrock client setup
aws_client = boto3.client(service_name="bedrock-runtime")
llm = BedrockChat(
    client=aws_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "max_tokens": 768,
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

    query = fquery = f"""
Summarize Reddit feedback on {restaurant_name} in 3–5 short bullets focusing on food, service, and value.
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
    
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dish keywords from text file
with open("dish_keywords.txt", "r") as f:
    dish_keywords = [line.strip().lower() for line in f if line.strip()]

# Initialize matcher with lowercase dish patterns
dish_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
dish_matcher.add("DISH", [nlp.make_doc(dish) for dish in dish_keywords])

import re
from collections import Counter

def extract_dish_mentions(posts):
    all_text = " ".join(post["title"].lower() for post in posts)
    mentions = [dish for dish in dish_keywords if re.search(rf"\b{re.escape(dish)}\b", all_text)]
    return Counter(mentions).most_common()
