# Restaurant Reddit Sentiment Dashboard

A Streamlit-powered dashboard that summarizes customer sentiment and reviews for restaurants based on recent Reddit posts. Built using AWS Bedrock (Claude 3 Sonnet), LangChain, and VADER Sentiment Analysis.

## Features

- Real-time Reddit post search
- AI-generated summaries with Claude 3 Sonnet via AWS Bedrock
- Average sentiment scoring with VADER
- Optional dish-level NLP extraction
- Filters for city, local restaurants, and fast food

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Environment Setup

Create a `.env` file with the following keys:

```env
# AWS Bedrock (optional if using profile-based auth)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-west-2

# Reddit API
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-secret
REDDIT_USER_AGENT=gsb570-agent
```

Alternatively, configure via AWS CLI:

```bash
aws configure --profile gsb570
```

## Running the App

```bash
streamlit run app.py
```

The app will launch at http://localhost:8501.

## AI Model: Claude 3 via Bedrock

This project uses:

```
anthropic.claude-3-sonnet-20240229-v1:0
```

Make sure:
- Your IAM user has `AmazonBedrockFullAccess`
- Your region is `us-west-2`

## Project Layout

```
.
├── dashboard.py            # Streamlit app
├── reddit_sentiment.py     # Sentiment & Reddit helpers
├── requirements.txt
├── .env
└── README.qmd
```
