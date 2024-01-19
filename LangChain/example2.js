import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { config } from "dotenv";

config();

const chatModel = new ChatOpenAI({
  openAIApiKey: process.env["OPENAI_API_KEY"],
});

async function main() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a world class mathematician that convert series into formulas",
    ],
    ["user", "{input}"],
  ]);

  const chain = prompt.pipe(chatModel);

  const result = await chain.invoke({
    input: "1,2,3,5,8,13",
  });

  console.log(result.text);

  console.log("----------------");

  const result2 = await chain.invoke({
    input: "2,4,6,8,10",
  });

  console.log(result2.text);
}

main();
