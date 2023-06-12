# vector-index-text-generation
Create short blog posts that make use of given resources. This repository only uses PDF sources, but a full list of supported loaders can be found here: `https://python.langchain.com/en/latest/modules/indexes/document_loaders.html`.

# Getting started
`git clone https://github.com/vpaul18/vector-index-text-generation && cd vector-index-text-generation`

`pip install -r requirements.txt`

There is an already uploaded pdf, genomics.pdf. To run the tests on the given pdf:

Start the flask server:


`` flask run --debug``

Change the `HOST` vaiables to your flask server's address (http://127.0.0.1:5000 if you're running locally).


Change the `os.environ["OPENAI_API_KEY"]` to your OpenAI API key in app.py.

Run tests:


``cd test``

Upload the already present pdf:


`python3 upload.py`


Embed the pdf:


`python3 embed.py`


Create a short blog post:


`python3 create.py`

