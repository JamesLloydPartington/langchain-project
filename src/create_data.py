# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import asyncio
from langchain.output_parsers import PydanticOutputParser
from pydantic import Field

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from pathlib import Path

import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)



load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("OPEN_AI_API_KEY is not set")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")


DATA_PATH = Path("../data")


# first name, last name, phone number, email address, address, and biography.
class Account(BaseModel):
    first_name: str = Field(description="First name of the person")
    last_name: str = Field(description="Last name of the person")
    phone_number: str = Field(description="Phone number of the person")
    email_address: str = Field(description="Email address of the person")
    address: str = Field(description="Address of the person (where they were born or grew up)")
    biography: str = Field(description="Biography of the person (roughly 500 words)")

    def __str__(self):
        return f"""First Name: {self.first_name}\nLast Name: {self.last_name}\nPhone Number: {self.phone_number}\nEmail Address: {self.email_address}\nAddress: {self.address}\nBiography: {self.biography}"""

class Person(BaseModel):
    first_name: str
    last_name: str

class People(BaseModel):
    people: list[Person]

class AccountCreator:

    def __init__(self):
        pass
    
    @staticmethod
    async def create_famous_person_list(person_types: list[str], people_per_type: int) -> dict[str, People]:

        template = """
        Create {n_people} random famous people who are person_type:'{person_type}'.

        {format_instructions}

        YOUR RESPONSE:
        """

        parser = PydanticOutputParser(pydantic_object = People)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["n_people", "person_type"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        tasks = []
        for person_type in person_types:
            tasks.append(
                llm.apredict(
                    prompt_template.format(n_people=people_per_type, person_type=person_type)
                )
            )

        async_results = await asyncio.gather(*tasks)

        results: dict[str, People] = {}
        for i, result in enumerate(async_results):
            try:
                results[person_types[i]] = parser.parse(result)
            except Exception as e:
                results[person_types[i]] = []
                logger.warning(f"Error parsing result for person_type: {person_types[i]} because of error: {e}")

        return results

    @staticmethod
    async def create_accounts(people: People, retry_index: int = 0, retires: int = 3) -> list[Account]:
        template = """
        Here is a person named {person}.
        {person}
        
        Create a fake account for this person with the following information:
        {format_instructions}
        YOUR RESPONSE:
        """

        parser = PydanticOutputParser(pydantic_object = Account)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["person"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        tasks = []
        for person in people.people:
            tasks.append(
                llm.apredict(
                    prompt_template.format(person=person.dict())
                )  
            )
        
        async_results = await asyncio.gather(*tasks)


        index_of_failed_results = []
        results: list[Account] = []
        for i, result in enumerate(async_results):
            try:
                results.append(parser.parse(result))
            except Exception as e:
                index_of_failed_results.append(i)
                logger.warning(f"Error parsing result for person: {people.people[i]} because of error: {e}")

        if len(index_of_failed_results) > 0 and retry_index < retires:
            failed_people = People(people=[people.people[i] for i in index_of_failed_results])
            results.extend(await AccountCreator.create_accounts(failed_people, retry_index=retry_index+1, retires=retires))
        elif len(index_of_failed_results) > 0:
            logger.warning(f"Could not create accounts for {len(index_of_failed_results)} people")

        return results
    
    @staticmethod
    async def create_famous_people_accounts(person_types: list[str], people_per_type: int) -> list[Account]:
        famous_people_list = await AccountCreator.create_famous_person_list(person_types, people_per_type)

        all_people = People(people=[])
        for people in famous_people_list.values():
            all_people.people.extend(people.people)
        
        return await AccountCreator.create_accounts(all_people)
        

# articles with headlines, body, url, publication date, source and author.
class Article(BaseModel):
    headline: str = Field(description="Headline of the article")
    body: str = Field(description="Body of the article (roughly 500 words)")
    url: str = Field(description="URL of the article (a slug of the headline with a real domain)")
    publication_date: str = Field(description="Publication date of the article (from a random date between 2019-2023)")
    source: str = Field(description="Source of the article (use a different real news source domain with another random slugs)")
    author: str = Field(description="Author of the article (use a random name)")

    def __str__(self):
        return f"""{self.headline}\n{self.body}\nURL: {self.url}\nPublication Date: {self.publication_date}\nSource: {self.source}"""

class ArticleHeadline(BaseModel):
    headline: str = Field(description="Headline of the article")

class ArticleHeadlines(BaseModel):
    headlines: list[ArticleHeadline]

class ArticleCreator:

    def __init__(self):
        pass

    @staticmethod
    async def create_article_headlines(article_types: list[str], articles_per_type: int) -> dict[str, ArticleHeadlines]:

        template = """
        Create {n_articles} random spesific article headlines for article_type:'{article_type}'.

        {format_instructions}

        YOUR RESPONSE:
        """

        parser = PydanticOutputParser(pydantic_object = ArticleHeadlines)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["n_articles", "article_type"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        tasks = []
        for article_type in article_types:
            tasks.append(
                llm.apredict(
                    prompt_template.format(n_articles=articles_per_type, article_type=article_type)
                )
            )

        async_results = await asyncio.gather(*tasks)

        results: dict[str, ArticleHeadlines] = {}
        for i, result in enumerate(async_results):
            try:
                results[article_types[i]] = parser.parse(result)
            except Exception as e:
                results[article_types[i]] = []
                logger.warning(f"Error parsing result for article_type: {article_types[i]} because of error: {e}")

        return results
    
    @staticmethod
    async def create_articles(article_headlines: ArticleHeadlines, retry_index: int = 0, retires: int = 3) -> list[Article]:
        template = """
        Here is an article headline:
        {headline}

        Create an article for this headline with the following information:
        {format_instructions}
        YOUR RESPONSE:
        """

        parser = PydanticOutputParser(pydantic_object = Article)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["headline"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        tasks = []
        for headline in article_headlines.headlines:
            tasks.append(
                llm.apredict(
                    prompt_template.format(headline=headline.dict())
                )  
            )
        
        async_results = await asyncio.gather(*tasks)


        index_of_failed_results = []
        results: list[Article] = []
        for i, result in enumerate(async_results):
            try:
                results.append(parser.parse(result))
            except Exception as e:
                index_of_failed_results.append(i)
                logger.warning(f"Error parsing result for headline: {article_headlines.headlines[i]} because of error: {e}")

        if len(index_of_failed_results) > 0 and retry_index < retires:
            failed_headlines = ArticleHeadlines(headlines=[article_headlines.headlines[i] for i in index_of_failed_results])
            results.extend(await ArticleCreator.create_articles(failed_headlines, retry_index=retry_index+1, retires=retires))
        elif len(index_of_failed_results) > 0:
            logger.warning(f"Could not create articles for {len(index_of_failed_results)} headlines")

        return results
    
    @staticmethod
    async def create_articles_from_types(article_types: list[str], articles_per_type: int) -> list[Article]:
        article_headlines = await ArticleCreator.create_article_headlines(article_types, articles_per_type)

        all_headlines = ArticleHeadlines(headlines=[])
        for headlines in article_headlines.values():
            all_headlines.headlines.extend(headlines.headlines)
        
        return await ArticleCreator.create_articles(all_headlines)

def main():

    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    if not Path.joinpath(DATA_PATH, "accounts").exists():
        Path.joinpath(DATA_PATH, "accounts").mkdir()

    if not Path.joinpath(DATA_PATH, "articles").exists():
        Path.joinpath(DATA_PATH, "articles").mkdir()

    accounts = asyncio.run(AccountCreator.create_famous_people_accounts(["actors", "athletes", "musicians", "scientists", "writers"], 10))
    for i, account in enumerate(accounts):
        # save the account to a file
        with open(Path.joinpath(DATA_PATH, "accounts", f"{i}.txt"), "w") as f:
            f.write(str(account))

    
    articles = asyncio.run(ArticleCreator.create_articles_from_types(["Physics", "AI/ML", "Quantum Computing", "Renewable Energy", "Healthcare"], 10))
    for i, article in enumerate(articles):
        # save the account to a file
        with open(Path.joinpath(DATA_PATH, "articles", f"{i}.txt"), "w") as f:
            f.write(str(article))


if __name__ == "__main__":
    main()