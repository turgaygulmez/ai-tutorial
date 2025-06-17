# OpenAI

  - [GPT-4o](#gpt-4o)
    - [Roles](#roles)
    - [Using data](#using-data)
    - [Understanding Tokens](#understanding-tokens)
  - [Embeddings](#embeddings)


OpenAI is a company that exposes their trained models through Rest API. The APIs are accessible to public and can be used with pay as you go pricing model. They also expose a public AI Tool (ChatGPT) which can be used for free. [Try it out!](https://chat.openai.com/)

What models are available on OpenAI?

- GPT-4o is a large multimodal model (accepting text or image inputs and outputting text) that can solve difficult problems
- DALL·E is a model that can generate and edit images given a natural language prompt
- TTS is a set of models that can convert text into natural sounding spoken audio
- Whisper is a model that can convert audio into text
- Moderation is a fine-tuned model that can detect whether text may be sensitive or unsafe
- Embeddings is a set of models that can convert text into a numerical form

Each models are available through APIs that can be used to perform a complex tasks.

They also provide a [Playground](https://platform.openai.com/playground?mode=chat) platform for their chat model with a very simple interface.

### GPT-4o

GPT-4o is a model that can perform both chat and completions tasks using the Chat Completions API.

Local development setup steps:

1. Install node18 or higher
2. Register on openAI platform
3. Create API key on [api keys page](https://platform.openai.com/api-keys)
4. Set an environment variable as "OPENAI_API_KEY" and set the api key you have generated
5. [Check simple OpenAI example](./samples/basic.js)

The example given above will ask question to OpenAI service "Who was the first president of USA?" and get the right answer once the service resolves our request.

We have not provided any data to OpenAI but yet it has answered our question correctly. The reason behind is, their models are already trained with very large amount of data. But that does not mean OpenAI will answer all the questions you ask to it.

For example if we ask OpenAI,

"What was the weather on 16/01/2024?" [See example](./samples/weather.js)

You will get an answer as such:

"I'm sorry, but as my last update was in March 2023 and I don't have the ability to....."

That is because OpenAI models are trained up until some dates. For instance gpt-4o has been trained up till October 2023

Before we jump into how to resolve this issue, lets first undertand the roles in Chat Completions API.

#### Roles

- User: Defines question
- System: Defines the system role and behavior
- Assistant: Models response depends on user question

```json
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "Knock knock."},
{"role": "assistant", "content": "Who's there?"},
{"role": "user", "content": "Orange."}
```

#### Using data

There are number of ways to use your own data

**Opt in to store data on OpenAI servers**

Opt in on openai server might be benefitial in case your data is not sensitive.

**Use assistants (endpoint) to store data**

The same thing goes with assistants but assistants endpoint might be pricy.

**Send data as part of the prompt**

For non-sensitive data, this might be the easiest way to do so. You can simply inject your data as part of your prompts. This can be done by defining data as part of user or system

See [Example](./samples/weather-v2.js)

**Use embeedings**

This is one of the most widely used option especially for large data. An embedding is a vector (a list of numbers) that represents an input object.
With embeedings you can find the relavent information and pass it along with your prompt.


Each method has its own advantages and disadvantages. Depending on the project and usecase one can be more benefitical than the other.


#### Understanding Tokens

Language models read and write text in chunks called tokens. In English, a token can be as short as one character or as long as one word (e.g., a or apple), and in some languages tokens can be even shorter than one character or even longer than one word.

For example, the string "ChatGPT is great!" is encoded into six tokens: ["Chat", "G", "PT", " is", " great", "!"].

The total number of tokens in an API call affects:

How much your API call costs, as you pay per token
How long your API call takes, as writing more tokens takes more time
Whether your API call works at all, as total tokens must be below the model’s maximum limit

Use this [link](https://platform.openai.com/tokenizer) to calculate the total token.

Its very important to undertand tokens when dealing with large data.

### Embeddings

An embedding is a vector representation of a piece of data (e.g. some text) that is meant to preserve aspects of its content and/or its meaning. Chunks of data that are similar in some way will tend to have embeddings that are closer together than unrelated data. OpenAI offers text embedding models that take as input a text string and produce as output an embedding vector. Embeddings are useful for search, clustering, recommendations, anomaly detection, classification, and more.

/v1/embeddings

An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

In a nutshell, the way embeeding works as such

1. First of all identify your data. For example list of customer reviews that contains following structure

```json
[
  { "userId": "1234", "content": "The product was amazing!" },
  { "userId": "2345", "content": "It was a disaster product. I am regretful" }
]
```

2. Call open AI embeddings api to get the embeedings for each and every reviews
3. Store them in a vector database. EX: HNSWLib memory database
4. Ideally store them in a file system to be able to reuse again
5. Load your embeedings with HNSWLib and perform actions EX: similaritySearch

There are lots of vector stores which can be used depending on your setup. Below are some of the vector database examples:

- Memory Vector Store
- Chroma
- Elasticsearch
- FAISS
- LanceDB
- Milvus
- MongoDB Atlas