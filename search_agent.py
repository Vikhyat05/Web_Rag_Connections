
""" 
Basic Functionality 
- Generate answers based on knowledge base content and provide sources for the answers.
- Gives additional content from the internet to deep dive for different content types (Podcast, YouTube, Articles).

3rd iteration of Continuous Search

- Adds citation from the knowledge base.
- Streams output from tools used by the agents.
- Responds to greetings like a normal chatbot.

"""


import os
from dotenv import load_dotenv
from listennotes import podcast_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from operator import itemgetter
import json
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import requests
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from langchain.agents import tool
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain_community.document_loaders import BraveSearchLoader



load_dotenv()

PODCAST_API_KEY = os.getenv("PODCAST_API_KEY")
api_key = PODCAST_API_KEY

brave_api = os.getenv("BRAVE_API")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# The API endpoint URL
url = "https://api.search.brave.com/res/v1/videos/search"

client = podcast_api.Client(api_key=api_key)



embeddings = OpenAIEmbeddings()
parser = StrOutputParser()
index_name = "ragwithsources"

vectorstore = PineconeVectorStore.from_existing_index(
    index_name, 
    embeddings,
)

retriever  = vectorstore.as_retriever(search_kwargs={"k": 10})



@tool("knowledge_chain")
def KnowledgeBase(query):
    """ Use this function to get answers for the users Knowledge base"""
    document_prompt = PromptTemplate.from_template("""

    source: {url} ,

    context: {content}
    """)

    document_chain = RunnableLambda(
        lambda document: {
            "content": document.page_content, 
            "url": document.metadata["source"]
        }
    ) | document_prompt

    retrieval_chain = retriever | document_chain.map() | (lambda docs: "\n".join([i.text for i in docs]))


    return (retrieval_chain.invoke(query))


class PodcastSearchInput(BaseModel):
    query: str = Field(description="Things or topic to search for")

@tool(args_schema=PodcastSearchInput) 
def podacast_search(query: str) -> dict:
    """ Search for podcast related to the topic"""
    response = client.search(q=query,)

    podcasts  = response.json()['results'][0:5]

    formatted_data = "\nSome Suggested podcasts are \n"
    for index, podcast in enumerate(podcasts, start=1):
        formatted_data += f"{index}. {podcast['title_original']}\n{podcast['link']}\n\n"

    return(formatted_data)



class ArticleSearchInput(BaseModel):
    query: str = Field(description="Query for which we need to search article for")

class VideoSearchInput(BaseModel):
    query: str = Field(description="Query for which we need to search videos for")

   
@tool(args_schema=ArticleSearchInput)
def ArticleSearch(query):

    """ Fetch the artciles from brave search based on the user query"""
    loader = BraveSearchLoader(
        query=query, api_key=brave_api, search_kwargs={"count": 3}
    )
    docs = loader.load()
    search_results = [doc.metadata for doc in docs]
    results = ""

    for i, video in enumerate(search_results, start=1):
        title = video.get('title', 'No Title')
        link = video.get('link', 'No Link')
        
        results += f"{i}. {title}\nsource: {link}\n\n"
        
    return(results)

@tool(args_schema=VideoSearchInput)
def VideoSearch(query):

    """Fetch the videos from brave search based on the user query """
    
    api_key = brave_api
    url = "https://api.search.brave.com/res/v1/videos/search"
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": api_key,  
    }

    
    params = {
        "q": query,  
    }

    
    response = requests.get(url, headers=headers, params=params)

    results = ""
    
    if response.status_code == 200:
        
        search_results = response.json()
        
        video_results = search_results.get('results', [])
        
        if video_results:
            
            for i,video in enumerate(video_results[:2],start=1):  
                title = video.get('title', 'No Title')
                video_url = video.get('url', 'No URL')
                results += (f"{i}. {title}\nLink: {video_url}\n\n")
        else:
            print("No video results found.")
    else:
        
        print(f"Failed to fetch search results: {response.status_code}")
        print(response.text)

    return(results)

search_model = ChatOpenAI(model = 'gpt-4-0125-preview')
p = ChatPromptTemplate.from_messages([
    ('system','Based on the user query create a another query which asks for podcasts, vidoes and articles on topics which user mentioned in the query '),
    ('user','{input}')
])

ain = p|search_model|StrOutputParser()

@tool("chain_tool")
def QueryModification(query):
    """Call this function when user need to get its query modified """
    x = {"input":query}
    return ain.invoke(x)


functions = [
    convert_to_openai_function(podacast_search),
    convert_to_openai_function(VideoSearch),
    convert_to_openai_function(QueryModification),
    convert_to_openai_function(KnowledgeBase),
    convert_to_openai_function(ArticleSearch)
    ]



from langchain.memory import ConversationBufferMemory

# search_model = ChatOpenAI(model = 'gpt-4-0125-preview').bind(functions = functions)

search_model = ChatOpenAI(model = 'gpt-4-0125-preview',
                          streaming= True,
                          callbacks=[StreamingStdOutCallbackHandler()])

memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")


# agent_executor.invoke({"input":"What is Gen AI"},return_only_outputs=True)['output']

### Historical Context
        #  - The concept of Generative AI, though a recent trend, has a history that goes back at least 70 years. It 
        #     began in the 1950s with the inception of text analytics and has evolved to include powerful language models like 
        #     GPT (Generative Pre-trained Transformer) - [Link1](~[source url link1]) [Link3](~[source url link3])


tools = [podacast_search,VideoSearch,ArticleSearch,QueryModification,KnowledgeBase]

# (Use all the urls links provided by the KnowledgeBase tool for citation)
prompt = ChatPromptTemplate.from_messages([
    ('system',""" 
    
    You are an helpful AI assitant. 
    
    Keep 7 points in mind when you are answering user quesrtions

    1. When every user asks for an information ot question you should always refer the knowledge base first.
    2. Give the detailed asnwers in a well strucured format, use headings and pointers.  
    3. IMPORTANT: Do citation for every content inside heading for each defination and pionter generated in the answer with the sources url links provided by the KnowledgeBase tool.
    Citation should be in the fomat like  [Link1](~[source url of link1]), [Link2](~[source url of link2). For same urls link number will be same
        
        >> Look at the examplew to see how to cite each line of the generated answers
        
        ### What is Gen AI
        Generative artificial intelligence (GenAI) is a type of artificial intelligence that is capable of creating various forms of content such as text, 
        images, videos, and other data in response to prompts. GenAI utilizes generative models to learn patterns and structures from input training data 
        and then generates new data based on this learning. This technology has seen significant advancements, particularly with 
        the rise of transformer-based deep neural networks, leading to the development of various generative AI systems 
        like chatbots (e.g., ChatGPT, Copilot), text-to-image systems (e.g., Stable Diffusion, DALL-E), and text-to-video generators (e.g., Sora) 
        by companies such as OpenAI, Microsoft, Google, and others  - [Link1](~[source url link1])
        
        ###Capabilities and Applications
        - Gen AI has limitations in achieving true creativity, it can create multiple forms of content across wide topics at speed and scale, 
        which can be leveraged across industries, functions, and personas to drive organizational goals. - [Link1](~[source link1]), [Link2](~[source url link2)
        - Generative AI can process vast amounts of data in any format, including text, audio, image, and video. Link1](~[source url link1)
        - It can work 24/7 to provide answers and create personalized content for users. - [Link2](~[source url link2)
        
        
    4.  After the KnowledgeBase tool's answer mention All Unique Sources Used after the answer givem using knowledge base as shown below
        All Sources Used 
        Link1: Complete link for link1 
        Link2: Complete link for link2
        Link3: Complete link for link3

        >> Note: Link numner should be unique for every source. Do not repeat link number (Link1 or Link2) for the same source link  

    5. Always suggest podcast, articles and video using the tools provided to dive deeper into the topic for every user's question  

        Give answer in the below format
        
        ### For further exploration, check out these resources:


        ####Podcasts
        - **Best of ThinkCast 2023: Generative AI** - [Listen here](~[http://thinkcast.libsyn.com/best-of-thinkcast-2023-generative-ai-generative-ai-and-more-generative-ai?utm_source=listennotes.com&utm_campaign=Listen+Notes&utm_medium=website](http://thinkcast.libsyn.com/best-of-thinkcast-2023-generative-ai-generative-ai-and-more-generative-ai?utm_source=listennotes.com&utm_campaign=Listen+Notes&utm_medium=website)~)
        - **343: Demystifying Generative AI** - [Listen here](~[https://www.forrester.com/cx-cast/343-demystifying-generative-ai/?utm_source=listennotes.com&utm_campaign=Listen+Notes&utm_medium=website](https://www.forrester.com/cx-cast/343-demystifying-generative-ai/?utm_source=listennotes.com&utm_campaign=Listen+Notes&utm_medium=website)~)

        #### Videos
        - **What is Generative AI? Everything You Need to Know** - [Watch here](~[https://www.youtube.com/watch?v=G2fqAlgmoPo](https://www.youtube.com/watch?v=G2fqAlgmoPo)~)

        #### Articles
        - **What is Generative AI? Everything You Need to Know** - [Read more](~[https://www.techtarget.com/searchenterpriseai/definition/generative-AI](https://www.techtarget.com/searchenterpriseai/definition/generative-AI)~)
        - **Google Generative AI – Google AI** - [Read more](~[https://ai.google/discover/generativeai/](https://ai.google/discover/generativeai/)~)
        - **Explained: Generative AI | MIT News** - [Read more](~[https://news.mit.edu/2023/explained-generative-ai-1109](https://news.mit.edu/2023/explained-generative-ai-1109)~)

    6. If there is no useful information in knowledge base 
    then say "There is no information about the topic in you knowledge base but according to me (add your response)"

    7. Give the answers for all the tools in markdown format
    """
     ),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user',"""{input} """),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])


from langchain.agents import AgentExecutor, create_openai_tools_agent

agent = create_openai_tools_agent(search_model, tools, prompt)

agent_executor  = AgentExecutor(agent=agent, tools=tools, verbose=False, memory=memory)
agent_executor.invoke({"input":"What is Gen AI"})['output']







