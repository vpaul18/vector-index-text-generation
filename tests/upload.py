"""
Test request that uploads the genomics.pdf
"""

import requests

# my dev server
HOST= "Your Flask Server"



# Open the file in binary mode
with open('../genomics.pdf', 'rb') as f:
    # Define the file to be sent
    files = {'file': ('genomics.pdf', f)}

    # Send the POST request to the upload route
    try:
        response = requests.post(
            f'{HOST}/upload', 
            files=files,
            timeout=5000
            )
    except Exception as e:
        print(e)

# Print the server's response
print(response.text)
