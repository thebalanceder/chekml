import webbrowser
import time

def search_datasets(keywords, search_engine="google"):
    search_urls = []
    
    # Base search URL (can be Google, Bing, DuckDuckGo, etc.)
    if search_engine == "google":
        base_url = "https://www.google.com/search?q="
    elif search_engine == "duckduckgo":
        base_url = "https://duckduckgo.com/?q="
    elif search_engine == "bing":
        base_url = "https://www.bing.com/search?q="
    else:
        base_url = "https://www.google.com/search?q="

    # Create search URLs for each keyword
    for keyword in keywords:
        query = f"{keyword} dataset download open source"
        url = base_url + query.replace(" ", "+")
        search_urls.append(url)

    # Open each search URL in the default browser (with delay to prevent overload)
    for url in search_urls:
        webbrowser.open(url)
        time.sleep(1)  # Pause between opening tabs

# Example keyword list (from LLM)
keywords = [
    "Urban Heat Island Data",
    "Temperature Datasets",
    "Urban Fabric Index",
    "Land Use Zonal Data",
    "Population Density Maps",
    "Building Heights Records",
    "Impervious Surfaces Map",
    "Thermal Comfort Indices",
    "Health Metrics Dataset",
    "Air Quality with Temperature"
]

# Search!
search_datasets(keywords)

