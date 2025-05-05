import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
import random
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import MessagesPlaceholder
import pandas as pd
import ast
from langchain.memory import ConversationBufferMemory

load_dotenv()
embeddings_model = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0)
parser = StrOutputParser()


# Simulating an Database content
df = pd.read_csv('output.csv')
df['Content_Tag'] = df['Content_Tag'].apply(lambda x: ast.literal_eval(x))
df['Total_Tags'] = df.apply(lambda row: row['Content_Tag'] + [row['Main_Tag']], axis=1)

class Knowledge_Tags_input(BaseModel):
    query: str = Field(description="Topic or theme to search tags for")

@tool(args_schema=Knowledge_Tags_input) 
def specific_knowledge_tags(query):

    """ 
    This function fetches type of tags and content present in the knowledge base realted to the given topic or theme
    """
    
    # Flatten the list of lists and get unique tags directly as a set
    unique_tags_set = set(tag for sublist in df['Total_Tags'] for tag in sublist)

    # Create a string of quoted strings directly from the set
    string_of_strings = ', '.join(f'"{s}"' for s in unique_tags_set)

    doc = Document(page_content=string_of_strings)

    tag_prompt1 = ChatPromptTemplate.from_template("""
    Tags: {Tags}
    Query:{Query}
    You are given Tags which represent content in the knowledge base. For the given user query you need to give all the tags related to the query. 
    Include the tags in the answer even if they are closely related. 

    Give only those tags which are in the list and nothing else
    """)

    # tag_prompt1.invoke({"Tags":doc.page_content,"Query":"What type of content do I have related to AI"})
    chain = tag_prompt1|model|parser
    return(chain.invoke({"Tags":doc.page_content,"Query":query}))
class Connection_tags_input(BaseModel):
    tags1: list = Field(description="List of tags related to one of the topic which will be used to form connection")
    tags2: list = Field(description="List of tags related to second topic which will be used to form connection")

# Tool to fetch content that helps create connections between two topics
@tool(args_schema=Connection_tags_input)
def Connections(tags1, tags2):
    """ 
    Use this function to get the content which can be used to created connections between two topics
    """
    text = ""

    # Filter content based on tags1
    tags_to_filter = tags1
    filtered_df1 = df[df['Total_Tags'].apply(lambda tags: any(tag in tags for tag in tags_to_filter))]
    filtered_df1 = filtered_df1.reset_index()

    # Sample two entries randomly from filtered content
    try:
        random_indices = random.sample(range(0, len(filtered_df1)), 2)
    except:
        random_indices = [0]
    
    # Append relevant source and summary to result string
    for i in random_indices:
        text += "\n \n Source: " + filtered_df1['Source'][i] + "\n \n Content: " + filtered_df1['Summary'][i]

    # Repeat filtering for tags2
    tags_to_filter = tags2
    filtered_df2 = df[df['Total_Tags'].apply(lambda tags: any(tag in tags for tag in tags_to_filter))]
    filtered_df2 = filtered_df2.reset_index()

    try:
        random_indices = random.sample(range(0, len(filtered_df2)), 2)
    except:
        random_indices = [0]

    for i in random_indices:
        text += "\n \n Source: " + filtered_df2['Source'][i] + "\n \n Content: " + filtered_df2['Summary'][i]

    return text

# Convert the tools into OpenAI-compatible function format
functions = [
    convert_to_openai_function(specific_knowledge_tags),
    convert_to_openai_function(Connections),
]

# Initialize streaming model for live output
search_model = ChatOpenAI(
    model='gpt-4-0125-preview',
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Use buffer memory to maintain chat history
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# List of available tools for the agent
tools = [specific_knowledge_tags, Connections]

# Define the system prompt and interaction flow
prompt = ChatPromptTemplate.from_messages([
    ('system',""" 
    You are an helpful assistant. Your have 2 jobs, one is to help the users get details about the their knowledge base and secoond is to help them form 
    connection and generate insights between any two topics which they choose

    << How to Give detail about their Knowledge base>>
    1. User will give you the query telling which type of content availability they want to from their knowledge base. 
    For eg: What type content do i have related to space ?
    2. You will use "specific_knowledge_tags" tool provided to get the content related to space in their knowledge base. Give one line description for each tag if possible 
    3. If their is no content available realted to user query then simply reply "There is no content related to space in your knowledge base"

    << How to Form connection between two topics given by user
    1. For eg user ask: "Form connection between AI and Agriculture" 
    2. You seperate the topic between which conection needs to be formed in this AI and Agriculture
    3. Then you will call the "specific_knowledge_tags" tool for each  topic to get the relevant tags. Do not change the spelling of tags at all keep them as it is
    4. Create the list of tags for each topic and send them as input to "Connections" tool
    5. "Connections" Toll will provide with the content for bith the topics with the source of the content
    6. Understand the core idea of the topic using the content and then form the relationshio between the topic and give innovative insight or idea
    generated by forming relationship bwteen the topics. Keep the answer limited to the content received 
    7. At the end mention all the sources urls and named from both the topic used to form connection.
        >> Look at example below for refrence 
        
        Agriculture 
        - Source 1 [~url link]
        - Source 2 [~url link]

        AI
        - Source 1 [~url link]
        - Source 2 [~url link]
     """
     ),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user',"""{input} """),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

# Create the agent using tools and prompt
from langchain.agents import AgentExecutor, create_openai_tools_agent

agent = create_openai_tools_agent(search_model, tools, prompt)

# Bind the agent and tools into a final executor with memory
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, memory=memory)

# Loop to accept user queries and invoke agent responses
while True:
    query = input("\nUser: ")
    agent_executor.invoke({"input": query})['output']
    
    if query.lower() == "bye":
        break

