import { ChatPromptTemplate } from "@langchain/core/prompts";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { getChatModel } from "./core.js";

async function run() {
  const chatModel = getChatModel();

  const embeddings = new OpenAIEmbeddings();

  const vectorStore = await HNSWLib.load("./localvector", embeddings);

  const prompt =
    ChatPromptTemplate.fromTemplate(`You are given a context which contains APPLE stock prices with its respective date:

  <context>
  {context}
  </context>
  
  Question: {input}`);

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  const retriever = vectorStore.asRetriever({
    k: 50,
    filter: (item) => {
      return item.metadata.year === 2024;
    },
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  const result = await retrievalChain.invoke({
    input: "Based on the price given, how apple price is doing on 2024?",
  });

  console.log(result.answer);
}

run();
