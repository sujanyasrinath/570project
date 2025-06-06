import streamlit as st
from dotenv import load_dotenv
from reddit_sentiment import (
    get_reddit_post_titles_and_links,
    label_sentiment,
    get_sentiment_for_restaurant,
    search_reddit  # ✅ ADD THIS
)
import boto3
from langchain_aws import ChatBedrock
#from langchain_community.chat_models import BedrockChat
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import pandas as pd
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
#from langchain_community.tools.reddit_search.tool import RedditSearchRun

load_dotenv()

aws_client = boto3.client(service_name="bedrock-runtime")
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-west-2",  # your AWS region
    streaming=False,          # ✅ DISABLE STREAMING
    model_kwargs={
        "max_tokens": 768,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 0.9,
        "stop_sequences": ["\n\nHuman"]
    }
)

prompt = hub.pull("hwchase17/structured-chat-agent")

@st.cache_data(show_spinner="Fetching Reddit summary...")
def summarize_restaurant_reddit(restaurant_name, time_filter="month"):
    reddit_tool = Tool.from_function(
    func=lambda query: search_reddit(query, time_filter=time_filter),
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

def query_restaurant_feedback(restaurant_name: str, user_question: str):
    prompt = f"""
Use recent Reddit posts about **{restaurant_name}** to answer the following question:

**{user_question}**

✅ Base your answer only on actual Reddit user discussions.
❌ Do NOT guess or make up facts. Be concise and helpful.

Reply in 2–4 sentences.
"""
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error generating response: {e}"

st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")

st.sidebar.title("Reddit Sentiment Explorer")
restaurant_query = st.sidebar.text_input("Enter Restaurant Name", "")
location_filter = st.sidebar.text_input("Optional: Enter City or Location", "")
time_filter = st.sidebar.selectbox("Time Range for Reddit Posts", ["month", "year", "all"])
exclude_chain = st.sidebar.checkbox("Favor Local/Independent Restaurants Only")
fast_food_only = st.sidebar.checkbox("Filter to Fast Food Mentions Only")

st.markdown("""
    <h1 style='color:#ff6d01; font-weight:bold; font-size:48px;'>Restaurant Reddit Sentiment</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='font-size:36px;'>Reddit Insights</h2>
""", unsafe_allow_html=True)

if restaurant_query:
    full_query = f"{restaurant_query} {location_filter}" if location_filter else restaurant_query
    if exclude_chain:
        full_query = f"{full_query} -McDonalds -Starbucks -Chick-fil-A -Taco Bell -Subway -Dominos"
    if fast_food_only:
        full_query = f"fast food {full_query}"

    with st.spinner(f"Analyzing Reddit discussions for {full_query}..."):
        reddit_result = summarize_restaurant_reddit(full_query, time_filter)
        st.markdown(f"### {reddit_result['restaurant']}")

        st.metric("Average Sentiment Score", f"{reddit_result['avg_sentiment_score']:.2f}",
                  label_sentiment(reddit_result['avg_sentiment_score']))

        st.markdown("#### Key Reddit Insights")
        summary_lines = reddit_result["summary"].split("\n")
        if summary_lines and "based on recent reddit posts" in summary_lines[0].lower():
            summary_lines = summary_lines[1:]

        for bullet in [line.strip("\u2022-* \n") for line in summary_lines if line.strip()]:
            st.markdown(f"- {bullet}")

        with st.expander("View Reddit Posts Used in Summary"):
            posts = get_reddit_post_titles_and_links(full_query)
            if not posts:
                st.markdown("_No Reddit posts found._")
            else:
                for post in posts:
                    st.markdown(f"- [{post['title']}]({post['url']})")

    st.markdown("### Ask the Assistant")
    user_question = st.text_input("What do you want to know about your customer feedback?")
    if user_question:
        with st.spinner("Generating AI response..."):
            answer = query_restaurant_feedback(full_query, user_question)
            st.success(answer)

    st.markdown("### Official Website")
    search_url = f"https://www.google.com/search?q={quote_plus(full_query + ' official website')}"
    st.markdown(f"🔗 [Search for {restaurant_query} website]({search_url})")
else:
    st.info("Please enter a restaurant name to view Reddit insights.")