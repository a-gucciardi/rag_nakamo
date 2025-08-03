from web_search import google_search

with open("google_config.txt", "r") as file:
    google_api_key, google_cx = [line.strip() for line in file.readlines()]

results = google_search("Regulatory in Medtech", google_api_key, google_cx)
for result in results:
    print(result["title"], result["link"])