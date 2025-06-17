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
  const result = await chatModel.invoke("Who was the first president of USA?");
  console.log(result.text);
}

main();
