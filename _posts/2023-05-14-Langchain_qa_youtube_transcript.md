---
title: Ask Dr.Berg with Langchain
date: 2023-05-04 12:00:00 -500
categories: [llm, tutorial]
tags: [langChain, ]
---

In this tutorial, we will ask Dr.Berg to answer health inquiries using Langchain.

We will start by explaining some basic concepts of LangChain library, then walk you through how to build the application.

### We are going to:
 - Extract youtube transcripts.
 - Chuck transcripts in smaller documents.
 - Create the embedding of the documents for semantic search.
 - Get the embeddings and then pass them over the Pinecone.
 - Get the documents that have the highest cosine similarity according to Pinecone.
 - Pass the documents to a Large Language Model (LLM) from OpenAI.
 - Run the chain and get the NL answer.

### What you need to know about LangChain Library
OpenAIEmbeddings
: Think of this as a way to compress data where we will search for data later on.

RecursiveCharacterTextSplitter
: This will split documents recursively by different characters - starting with "\n\n", then "\n", then " ". This is nice because it will try to keep all the semantically relevant content in the same place for as long as possible.

YoutubeLoader
: This will allow us to load the transcripts and metadata of a video given the id.

Picone
: Pinecone makes it easy to provide long-term memory for high-performance AI applications. 

## Ask Dr.Berg (videos)

Ask Dr.Berg (videos) is a demonstration of the retrieval-augmented question-answering application.

We use two of Dr.Berg's YouTube videos as a corpus. 

So someone can ask questions and get answers in the context of the videos. E.g., Why is sugar consumption unhealthy?

![jupyter notebook](/assets/lib/Langchain_qa_youtube_transcript/demo/demo_qa_youtube_langchain.png)

## Stack
We use LangChain to organize the Large Language model (LLM) invocation and prompt.

## Run it yourself
This application assumes you have prior knowledge to set up a virtual environment for Python. After activating your virtual environment, you are ready to install the dependencies.

### Install dependencies 
Go to the Task folder, and run the following
```bash
make environment
```
This will install the dependencies from the requirements.txt file.

### Create your .env file
You can take as a reference the .env.example file to create your .env file.

### Get your OpenAi and Pinecone Api_keys
It is a must to have the OPENAI_API_KEY to have access to OpenAI LLM models. 

![OpenAI - Api key](/assets/lib/Langchain_qa_youtube_transcript/openai/openai_api_key.png)

Go to Pinecone to get the PINECONE_API_KEY to store the embeddings in the cloud.

![Pinecone - environment](/assets/lib/Langchain_qa_youtube_transcript/pinecone/pinecone_api_key.png)

Note: Both third-party services offer a free trial, so there is no cost for experimenting with these awesome technologies.

Then, create an index in Pinecone.

![Pinecone - Create index](/assets/lib/Langchain_qa_youtube_transcript/pinecone/pinecone_create_index.png)


### Load libraries

```python
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader

import pinecone
import os
```

### Extract youtube transcripts

```python
json_path = "./data/videos.json"
with open(json_path) as f:
    video_infos = json.load(f)  
```

### Chuck transcripts in smaller documents

```python
base_url = "https://www.youtube.com/watch?v="
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

documents = []
for video_info in video_infos:
    video_id = video_info["id"]
    loader = YoutubeLoader.from_youtube_url(f"{base_url}{video_id}",add_video_info=True)
    transcript = loader.load()
    docs = text_splitter.split_documents(transcript)
    documents.extend(docs)
documents
```

This is the content of documents.
```text
[Document(page_content="warning this video may cause you to never want to eat sugar ever again so you might want to pause this video and get your last dose before I reveal some very interesting information and of course the more aware you are of what sugar does the less you're going to consume it I watch people all the time around me just consuming massive amounts of sugar and I'm like oh my gosh how could you possibly do that well it has to do with awareness they just don't know the effects now we probably all know the effects with weight gain and dental decay and fatty liver and high cholesterol and acne and high blood pressure increase candida infections and lowered immune system etc etc but there's some other things that I want to share with you deep inside your body that you need to know 

...

It gives you all sorts of great resources. I have all my YouTube videos on this app. And it's regularly uploading\nthe most recent ones. All the YouTube videos are also\nconverted to audio versions. Okay, so you can use\nit when you're walking, exercising, driving your car. I have a mini course on there. I'll be putting additional courses. I have a lot of recipes on there and this is new, and also PDF resources. So there's various downloads,\nPDFs that you can get as well. And if you wouldn't mind\nafter you download it, check it out, give me your unbiased review\nand tell me how you like it. I want to know.", metadata={'source': 'mRj1RKh4xyY', 'title': 'What Happens If You Stop Eating Sugar for 14 Days – Dr. Berg On Quitting Sugar Cravings', 'description': 'Unknown', 'view_count': 19043461, 'thumbnail_url': 'https://i.ytimg.com/vi/mRj1RKh4xyY/hq720.jpg', 'publish_date': '2018-12-31 00:00:00', 'length': 387, 'author': 'Dr. Eric Berg DC'})]
```
### Create the embedding of the documents for semantic search

```python
# create embeddings with OpenAI
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# change the string into a vector space that represent different documents
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
```

### Pinecone
We will store our embeddings in the cloud so that they can persist. Pinecone allows us to do that for free.
First, we need to create a new account to get the api_key and environment, so we initialize Pinecone.
Second, we must create an index for our vector with the following setup: Dimension: 1536 and Metric: Cosine.

```python
# init 
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV=os.getenv("PINECONE_ENV")

pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_ENV
)
index_name = "drberg"
```

### Get the embeddings and then pass them over the Pinecone

```python
docsearch =  Pinecone.from_texts([doc.page_content for doc in documents], embeddings, index_name=index_name)
```

### This are the documents that have the highest cosine similarity according to Pinecone

```python
query = "Why does sugar a risk for blood vessels?"
docs = docsearch.similarity_search(query)
docs
```
This is the content of docs.
```text
[Document(page_content="warning this video may cause you to never want to eat sugar ever again so you might want to pause this video and get your last dose before I reveal some very interesting information and of course the more aware you are of what sugar does the less you're going to consume it I watch people all the time around me just consuming massive amounts of sugar and I'm like oh my gosh how could you possibly do that well it has to do with awareness they just ...
```
This answer is to long. We need to provide a natural language answer.

### Get the answer in Natural Language (NL)

```python
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0, openai_api_key= OPENAI_API_KEY)
chain =  load_qa_chain(llm, chain_type="stuff")

query = "Why does sugar a risk for blood vessels?"
docs = docsearch.similarity_search(query)
```

### Run the chain and get the NL answer 

```python
chain.run(input_documents=docs, question = query)
```
Dr. Berg's final Answer:
```text
' Sugar can increase the risk for blood vessels because it can lead to insulin resistance, which blocks the absorption of nutrients, minerals, and vitamins. It can also lead to glycation, which is the damage of proteins in the blood, and can lead to mitochondrial dysfunction, which can cause oxidative stress and inflammation.'
```




