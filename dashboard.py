import streamlit as st
from dotenv import load_dotenv
from reddit_sentiment import (
    summarize_restaurant_reddit,
    get_reddit_post_titles_and_links,
    label_sentiment,
    get_sentiment_for_restaurant
)
import boto3
from langchain_community.chat_models import BedrockChat
from urllib.parse import quote_plus

@st.cache_data(show_spinner="Fetching Reddit summary...")
def cached_summary(restaurant_name):
    return summarize_restaurant_reddit(restaurant_name)

def query_restaurant_feedback(restaurant_name: str, user_question: str):
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
    prompt = f"""
Use recent Reddit posts about **{restaurant_name}** to answer the following question:

**{user_question}**

✅ Base your answer only on actual Reddit user discussions from the last month.
❌ Do NOT guess or make up facts. Be concise and helpful.

Reply in 2–4 sentences.
"""
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error generating response: {e}"

load_dotenv()
st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")

st.sidebar.title("Reddit Sentiment Explorer")
restaurant_query = st.sidebar.text_input("Enter Restaurant Name", "")

st.markdown("""
    <h1 style='color:#ff6d01; font-weight:bold; font-size:48px;'>Restaurant Reddit Sentiment</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='font-size:36px;'>Reddit Insights</h2>
""", unsafe_allow_html=True)

if restaurant_query:
    with st.spinner(f"Analyzing Reddit discussions for {restaurant_query}..."):
        reddit_result = cached_summary(restaurant_query)
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
            posts = get_reddit_post_titles_and_links(restaurant_query)
            if not posts:
                st.markdown("_No Reddit posts found._")
            else:
                for post in posts:
                    st.markdown(f"- [{post['title']}]({post['url']})")

    st.markdown("### Ask the Assistant")
    user_question = st.text_input("What do you want to know about your customer feedback?")
    if user_question:
        with st.spinner("Generating AI response..."):
            answer = query_restaurant_feedback(restaurant_query, user_question)
            st.success(answer)

    st.markdown("### Must-Try Dishes")
    with st.spinner("Identifying popular dishes from Reddit discussions..."):
        dish_query = "What are the must-try or most popular dishes people talk about at this restaurant?"
        dishes = query_restaurant_feedback(restaurant_query, dish_query)
        st.markdown(dishes)

    st.markdown("### Official Website")
    search_url = f"https://www.google.com/search?q={quote_plus(restaurant_query + ' official website')}"
    st.markdown(f"🔗 [Search for {restaurant_query} website]({search_url})")
else:
    st.info("Please enter a restaurant name to view Reddit insights.")
