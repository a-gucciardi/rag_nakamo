import requests

def google_search(query, api_key, search_engine_id, num_results=10):
    """
    Google Custom Search query.

    Args:
        query (str): The search query.
        api_key (str): Your Google API key.
        search_engine_id (str): Your Custom Search Engine ID.
        num_results (int): Number of search results to return.

    Returns:
        list: A list of search result dictionaries.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": search_engine_id,
        "num": num_results
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("items", [])
