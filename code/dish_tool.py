from langchain.tools import tool

@tool
def analyze_top_dishes(location: str, cuisine: str) -> str:
    """
    Extracts top dishes from reviews for businesses in the given location and cuisine.
    Uses NER and sentiment analysis to determine highly rated items.
    """
    # Placeholder – to be filled with the logic using NLP and sentiment
    return f"Top dishes for {cuisine} restaurants in {location} include: garlic naan, chicken tikka masala, and mango lassi."
