import { JSONLoader } from "langchain/document_loaders/fs/json";
import { config } from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

config();

async function storeVector() {
  const loader = new JSONLoader("./price.json");
  const docs = await loader.load();
  const metaDocs = docs.map((x) => {
    return {
      ...x,
      metadata: {
        ...x.metadata,
        year: new Date(JSON.parse(x.pageContent).date).getFullYear(),
      },
    };
  });

  const embeddings = new OpenAIEmbeddings();

  const vectorStore = await HNSWLib.fromDocuments(metaDocs, embeddings);

  // save vector for later usage
  vectorStore.save("./localvector");

  const items2024 = vectorStore.similaritySearchWithScore("get 2024 prices");

  console.log(items2024);
}

storeVector();
