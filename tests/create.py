"""
Test request that creates a short blog post given a topic
"""
import requests


# my dev server
HOST= "Your Flask Server"


# Define the headers for the request
headers = {
    'Content-Type': 'application/json',
}

# Define the data for the request.
data = {
    'topic': 'Genomics of cold adaptations in the Antarctic notothenioid fish radiation',
    'model': 'davinci', #here you can use davinci too; although gpt4 is preferred(takes longer)
    #'collection_name': 'your_collection_name'
}


# Send the POST request
response = requests.post(
    f'{HOST}/create', 
    headers=headers, 
    json=data,
    timeout=50000
    )

# Print the server's response
print(response.text)


