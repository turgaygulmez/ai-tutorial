import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";

config();

const chatModel = new ChatOpenAI({
  openAIApiKey: process.env["OPENAI_API_KEY"],
});

async function main() {
  const result = await chatModel.invoke("Who was the first president of USA?");

  console.log(result.text);
}

main();
