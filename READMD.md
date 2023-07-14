Project objective:

First step is to generate some synthetic data. We suggest two types:

- 50 account records covering first name, last name, phone number, email address, address, and biography.

- 50 technology articles with headlines, body, url, publication date, source and author.

After making that data, we’d like you to ingest it using Llama Index.

After that, we’d like you to make a LangChain agent which has tools enabling search over both the account and article indexes and is able to select the appropriate one based on the interaction with the user.

For example, you should be able to ask it: “Please find me the account for John Doe and give me his biography and address", “Please find me some articles about Artificial Intelligence” or “Who is the author of that?”.