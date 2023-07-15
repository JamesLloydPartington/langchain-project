from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import os
import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from pathlib import Path

import logging

import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("OPEN_AI_API_KEY is not set")
else:
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


PWD = Path(__file__).parent
DATA_PATH = Path.joinpath(PWD, "../data")

ACCOUNT_PATH = Path.joinpath(DATA_PATH, "accounts")
ARTICLE_PATH = Path.joinpath(DATA_PATH, "articles")

ACCOUNT_INDEX_PATH = Path.joinpath(DATA_PATH, "account_index")
ARTICLE_INDEX_PATH = Path.joinpath(DATA_PATH, "article_index")

if not ACCOUNT_INDEX_PATH.exists():
    ACCOUNT_INDEX_PATH.mkdir()

if not ARTICLE_INDEX_PATH.exists():
    ARTICLE_INDEX_PATH.mkdir()



def main():
    # article_documents = SimpleDirectoryReader(ARTICLE_PATH).load_data()
    # article_index = VectorStoreIndex.from_documents(documents=article_documents)
    # article_index.storage_context.persist(persist_dir=ARTICLE_INDEX_PATH)

    # account_documents = SimpleDirectoryReader(ACCOUNT_PATH).load_data()
    # account_index = VectorStoreIndex.from_documents(documents=account_documents)
    # account_index.storage_context.persist(persist_dir=ACCOUNT_INDEX_PATH)

    account_storage_context = StorageContext.from_defaults(persist_dir=ACCOUNT_INDEX_PATH)
    account_index = load_index_from_storage(account_storage_context)
    
    article_storage_context = StorageContext.from_defaults(persist_dir=ARTICLE_INDEX_PATH)
    article_index = load_index_from_storage(article_storage_context)


    tools = [
        Tool(
            name="account_index",
            func=lambda q: str(account_index.as_query_engine().query(q)),
            description="Used to search the user details name, email, phone number, address biographies.",
            return_direct=True,
        ),

        Tool(
            name="article_index",
            func=lambda q: str(article_index.as_query_engine().query(q)),
            description="Used to search the article titles, content, website urls, source and authors.",
            return_direct=True,
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm = ChatOpenAI(temperature=0)
    agent_executor = initialize_agent(
        tools, llm, agent="conversational-react-description", memory=memory
    )

    x = agent_executor.run(input="What article would Isaac Newton like to read from our articles")

    print(x)



if __name__ == "__main__":
    main()