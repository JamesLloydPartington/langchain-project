from typing import Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import os
import openai
from langchain.output_parsers import PydanticOutputParser

from llama_index import StorageContext, load_index_from_storage
from pathlib import Path
from llama_index.indices.base import BaseIndex

from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


import logging

import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()




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



class Account(BaseModel):
    first_name: str = Field(description="First name of the person")
    last_name: str = Field(description="Last name of the person")
    phone_number: str = Field(description="Phone number of the person")
    email_address: str = Field(description="Email address of the person")
    address: str = Field(description="Address of the person (where they were born or grew up)")
    biography: str = Field(description="Biography of the person (roughly 500 words)")

    def __str__(self):
        return f"""First Name: {self.first_name}\nLast Name: {self.last_name}\nPhone Number: {self.phone_number}\nEmail Address: {self.email_address}\nAddress: {self.address}\nBiography: {self.biography}"""



class CustomTool:

    def __init__(self,
        function: callable,
        function_name: str,
        description: str,
        input_model: BaseModel,
        output_model: BaseModel,
    ):
        self.function = function
        self.function_name = function_name
        self.description = description
        self.input_model = input_model
        self.output_model = output_model

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
    
    def describe(self, 
        show_function_name: bool = True,
        show_description: bool = True,
        show_input_model: bool = True,
        show_output_model: bool = True,
        ) -> str:

        description = "-"*10 + "\n"
        if show_function_name:
            description += f"function_name: '{self.function_name}'\n"
        if show_description:
            description += f"description: {self.description}\n"
        if show_input_model:
            description += f"input_model: {self.input_model.schema_json()}\n"
        if show_output_model:
            description += f"output_model: {self.output_model.schema_json()}\n"
        description += "-"*10 + "\n"
        return description

class Instruction:
    def __init__(self, text: str, tool: CustomTool):
        self.text = text
        self.tool = tool

    def run(self, llm: OpenAI, query: str, context: str):

        # turn the context into the tool's input model

        template = '''
        Given the original query:
        {query}

        Given the following context:
        {context}

        We are now want to run the following instruction:
        {instruction}

        Given the following information, put the following information into the input model:
        {input_model}

        Response:
        '''

        parser = PydanticOutputParser(pydantic_object = self.tool.input_model)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["context" , "query", "instruction"],
            partial_variables = {"input_model": parser.get_format_instructions()},
        )

        logger.info(f"Running instruction: {self.text}")

        template_str = prompt_template.format(context=context, query=query, instruction=self.text)

        logger.info(f"Template String: {template_str}")

        input_model_str = llm.predict(template_str)

        logger.info(f"Input Model: {input_model_str}")

        # turn the input model into the output model
        try:
            input_model = parser.parse(input_model_str)
        except Exception as e:
            logger.debug(f"Error parsing input model: {input_model_str}")
            logger.error(f"Error parsing input model: {e}")
            raise
        
        result = self.tool(input_model)

        logger.info(f"Result: {result}")

        return str(result)
    

class DeterministicPlannerAndExecutor:
    '''
    Given a query, this planner will return a list of instructions to execute. Cannot handle dynamic workflows.
    '''
    def __init__(self, tools: list[CustomTool], llm: OpenAI):
        self.tools = tools
        self.llm = llm

        self.tools_by_name = {tool.function_name: tool for tool in self.tools}

    def show_tools(self) -> str:
        description = ""
        for tool in self.tools:
            description += tool.describe()
        return description


    def create_plan(self, query: str) -> list[Instruction]:

        class InstructionOutput(BaseModel):
            action: str = Field(description="A single action to take with a tool. Suggest what needs to go into the tool.")
            function_name: str = Field(description="The name of the function to call (the function_name).")

        class PlanOutput(BaseModel):
            instructions: list[InstructionOutput] = Field(description="A list of instructions to execute which will complete the task.")
        
        template = '''
        Given the following tools:
        {tools}

        The following human query is:
        {query}


        Create a plan to execute the query. A plan is a list of instructions,
        an instruction is a single action to take using exactly one tool (exactly one function_name).
        In the action be precise about what you are looking for. 
        Make the instructions will acomplish all the tasks in the query.
        Provide a list of actions using tools in the format of:
        {instruction}

        Response:
        '''

        parser = PydanticOutputParser(pydantic_object = PlanOutput)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["tools", "query"],
            partial_variables={"instruction": parser.get_format_instructions()},
        )

        instructions_str = self.llm.predict(prompt_template.format(tools=self.show_tools(), query=query))

        print('-'*100)
        print(instructions_str)
        print('-'*100)

        try:
            instructions = parser.parse(instructions_str)
        except Exception as e:
            logger.error(f"Error parsing instructions: {e}")
            raise
        
        instruction_list: list[Instruction] = []
        for instruction in instructions.instructions:
            tool = self.tools_by_name[instruction.function_name]
            instruction_list.append(Instruction(text=instruction.action, tool=tool))

        return instruction_list
    
    def predict(self, query: str):
        instructions = self.create_plan(query=query)

        for instruction in instructions:
            logger.info('-'*100)
            logger.info(f"Running instruction: {instruction.text}")
        
        instruction_outputs = ""

        for i, instruction in enumerate(instructions):

            instruction_output = instruction.run(llm=self.llm, query=query, context=instruction_outputs)

            instruction_outputs += f"Output for instruction {i}:\n {instruction_output}\n"

        prompt = f'''
        Given the original query:
        {query}

        Given the following outputs for subsequent instructions:
        {instruction_outputs}

        Give a final answer to all the questions asked in the query:
        '''

        llm_output = self.llm.predict(prompt)

        return llm_output

    


if __name__ == "__main__":

    class IndexInput(BaseModel):
        query: str = Field(description="A semantic query to execute against the index.")
        number_of_results: int = Field(description="The number of results to return in a single string.")

    class IndexOutput(BaseModel):
        result: str = Field(description="The result of the query.")
        

    account_storage_context = StorageContext.from_defaults(persist_dir=ACCOUNT_INDEX_PATH)
    account_index = load_index_from_storage(account_storage_context)
    
    article_storage_context = StorageContext.from_defaults(persist_dir=ARTICLE_INDEX_PATH)
    article_index = load_index_from_storage(article_storage_context)

    def account_query(input: IndexInput) -> IndexOutput:
        return IndexOutput(
            result=str(account_index.as_query_engine(similarity_top_k=input.number_of_results).query(input.query))
        )
    
    def article_query(input: IndexInput) -> IndexOutput:
        return IndexOutput(
            result=str(article_index.as_query_engine(similarity_top_k=input.number_of_results).query(input.query))
        )

    tools = [
        CustomTool(
            function=account_query,
            function_name="account_index.query",
            description="This queries the accounts of users in the database with a semantic query, and returns one or more accounts as a single string, information includes: name, email, phone number, address, and other information.",
            input_model=IndexInput,
            output_model=IndexOutput,
        ),

        CustomTool(
            function=article_query,
            function_name="article_index.query",
            description="This queries the articles in the database with a semantic query, and returns one or more articles as a single string, information includes: title, content, author, date, and url and source.",
            input_model=IndexInput,
            output_model=IndexOutput,
        ),
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-4")

    agent = DeterministicPlannerAndExecutor(tools=tools, llm=llm)

    result = agent.predict(query="Find 7 articles about physics. What are their titles. Who in our database would be interested in these articles?")

    print('-'*100)
    print(result)

    
        



# class CustomAgent():
#     def __init__(self, tools, llm, memory=None):


# def main():

#     account_storage_context = StorageContext.from_defaults(persist_dir=ACCOUNT_INDEX_PATH)
#     account_index = load_index_from_storage(account_storage_context)
    
#     article_storage_context = StorageContext.from_defaults(persist_dir=ARTICLE_INDEX_PATH)
#     article_index = load_index_from_storage(article_storage_context)


#     tools = [
#         # Tool(
#         #     name="account_index",
#         #     func=lambda q: str(account_index.as_query_engine().query(q)),
#         #     # description="Used to search the user details name, email, phone number, address biographies.",
#         #     description="This function does symantic search on the user's details name, email, phone number, address and biographies in the database. Provide a single string as input and it will return a json string of the user.",
#         #     return_direct=True,
#         # ),

#         Tool(
#             name="article_index",
#             func=lambda q: str(article_index.as_query_engine().query(q)),
#             description="This function does symantic search on the article which includes the title, content, author, date, url and source. This will return one or several articles as a single string.",
#             return_direct=True,
#         ),
#     ]

#     memory = ConversationBufferMemory(memory_key="chat_history")
#     llm = ChatOpenAI(temperature=0.5, model="gpt-4")

#     # x = llm.predict("What comes after monday?")

#     # print(x)
    
#     agent_executor = initialize_agent(
#         tools, llm, memory=memory
#     )

#     x = agent_executor.run(input="Find an article about black holes. What was the Source?")
#     print(x)

#     # llm = ChatOpenAI(temperature=0, model_name="gpt-4")
#     # model = ChatOpenAI(temperature=0, model_name="gpt-4")
#     # planner = load_chat_planner(model)
#     # executor = load_agent_executor(model, tools, verbose=True)
#     # agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

#     # x = agent.run("What is johnny depp's email in our database?")
    
#     # print(x)


# if __name__ == "__main__":
#     main()