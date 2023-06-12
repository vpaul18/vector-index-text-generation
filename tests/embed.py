"""
Test request that creates the embeddings & persists them
"""
import requests
import json

# my dev server
HOST= "Your Flask Server"



# Define the JSON payload
data = {
    'filename': 'genomics.pdf',  # The filename you uploaded in the upload request
    #'collection_name': 'your_collection_name'
}


# Send the POST request to the embed route
response = requests.post(
     f'{HOST}/embed', 
     data=json.dumps(data),
     headers={'Content-Type': 'application/json'},
     timeout=10000
     )

# Print the server's response
print(response.text)
