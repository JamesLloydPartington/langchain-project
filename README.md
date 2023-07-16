The project objective:
Use Langchain + Llama Index to build a conversational agent which can search over two indices.
1. Account index with 50 records covering first name, last name, phone number, email address, address, and biography.
2. Technology articles index with 50 records with headlines, body, url, publication date, source and author.

The agent should be able to answer questions like:
1. “Please find me the account for John Doe and give me his biography and address”
2. “Please find me some articles about Artificial Intelligence”
3. “Who is the author of that?” (this should be follow up of the previous question)


# Installation
1. Create a virtual environment with python 3.10 or higher `python -m venv venv`
2. Activate the virtual environment `source venv/bin/activate`
3. Install the requirements `pip install -r requirements.txt`

# Environment Variables
1. Create a `.env` file in the root directory
2. Add OPENAI_API_KEY to the `.env` file

# Create synthetic data
The data is created using the src/create_data.py script.

The AccountCreator.create_famous_people_accounts method creates famous people of type X.
The default is 10 of each of the following: ["actors", "athletes", "musicians", "scientists", "writers"]

The ArticleCreator.create_articles_from_types method creates articles of type Y.
The default is 10 of each of the following: ["Physics", "AI/ML", "Quantum Computing", "Renewable Energy", "Healthcare"]

The script will index both the accounts and articles with the Llama Index. The documents and indices are stored in the data folder.

# Run the agent
The agent is run using the src/agent.py script.

The chatbot will ask you to enter a question and will respond with an answer. This is done repeatedly until you control-c to exit the script.

The chatbot remembers the previous questions and answers and is able to query both the account and article indices to answer a question.

For example:
'Find the headlines of 3 articles about AI/ML, who might be interested to read them?'

```
The headlines of 3 articles about AI/ML are: 
1. "Machine Learning: The Key to Predictive Analytics"
2. "Exploring the Future of AI in Healthcare"
3. "Understanding the Impact of AI on Job Market"

The people who might be interested in reading these articles are:
1. James Watson: His interests in innovative thinking, problem-solving skills, and mentoring young people align with the topics of these articles.
2. Nikola Tesla: His interest in the design of the modern alternating current (AC) electricity supply system and magnetic flux density could extend to AI and ML.
3. Scarlett Johansson: Her interest in reading and dominating the fake world with her talents could make her interested in the impact of AI on various fields.
```