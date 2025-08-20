import requests

def search_github_datasets(query):
    github_api_url = "https://api.github.com/search/repositories"
    
    params = {
        "q": f"{query} in:name,description",
        "sort": "stars",
        "order": "desc",
        "per_page": 10
    }

    response = requests.get(github_api_url, params=params)
    results = response.json()

    for repo in results["items"]:
        print(f"Repo: {repo['full_name']}")
        print(f"URL: {repo['html_url']}")
        print("-" * 40)

# Example query
search_github_datasets("urban heat island dataset")

