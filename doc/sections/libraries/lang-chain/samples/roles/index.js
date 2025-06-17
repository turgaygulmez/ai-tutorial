import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";

config();

export const getChatModel = () => {
  const chatModel = new ChatOpenAI({
    azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
    azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
    azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
    azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
  });

  return chatModel;
};

async function main() {
  const chatModel = getChatModel();

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
