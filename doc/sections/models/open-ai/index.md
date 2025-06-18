# OpenAI

  - [GPT-4o](#gpt-4o)
    - [Roles](#roles)
    - [Understanding Tokens](#understanding-tokens)
  - [Embeddings](#embeddings)
  - [Training](#training)


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

#### Understanding Tokens

The total number of tokens in an API call affects:

How much your API call costs, as you pay per token
How long your API call takes, as writing more tokens takes more time
Whether your API call works at all, as total tokens must be below the model’s maximum limit

Use this [link](https://platform.openai.com/tokenizer) to calculate the total token.

### Embeddings

OpenAI offers text embedding models that take as input a text string and produce as output an embedding vector.

/v1/embeddings

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


### Training

OpenAI’s base models like GPT-4o are pre-trained on large datasets and are powerful out-of-the-box. However, for many applications, you may want to **customize** or **fine-tune** these models to better suit your specific needs or domain. Below are the main ways to train or adapt OpenAI models:

#### 1. Fine-tuning with OpenAI API

OpenAI supports **fine-tuning** on some base models (primarily GPT-3.5 and GPT-4 variants depending on availability). Fine-tuning allows you to train a specialized version of a base model on your own dataset to improve performance on specific tasks.

* **How it works**: You upload a labeled dataset of prompt-completion pairs.
* **Use case**: Customizing the model to speak in a particular style, domain-specific language, or to better understand your data.
* **Example**: Fine-tuning a customer support chatbot to use your company’s terminology.
* **API docs**: [Fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)

#### 2. Prompt Engineering / Few-shot Learning

Instead of training the model weights, you can **craft prompts** with examples or instructions that guide the model’s behavior without retraining.

* **How it works**: Provide the model with context or examples in the prompt.
* **Use case**: Quickly adapting model behavior for new tasks or formatting.
* **Example**: Provide 3 example questions and answers before asking your actual question.
* **Advantage**: No training cost or time; instant changes.
* **Limitation**: Can increase token usage; not as precise as fine-tuning.

#### 3. Using Embeddings for Custom Search or Retrieval-Augmented Generation (RAG)

Rather than training the model itself, you can preprocess your data by converting it into embeddings and use those embeddings to retrieve relevant context during prompts.

* **How it works**: Generate embeddings for your documents and store them in a vector database.
* **Use case**: Build domain-specific knowledge bases, chatbots, or search engines that augment the model with relevant external knowledge.
* **Advantage**: No changes to the base model; works well with large datasets.
* **Example**: A legal document assistant that finds relevant clauses and passes them as context in prompts.