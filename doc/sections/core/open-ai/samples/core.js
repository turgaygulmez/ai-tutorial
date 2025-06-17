import OpenAI from "openai";
import { config } from "dotenv";

config();

export const getOpenAI = () => {
  const openai = new OpenAI({
    apiKey: process.env.AZURE_OPENAI_API_KEY,
    baseURL: `https://${process.env.AZURE_OPENAI_API_INSTANCE_NAME}.openai.azure.com/openai/deployments/${process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME}`,
    defaultQuery: {
      "api-version": process.env.AZURE_OPENAI_API_VERSION,
    },
    defaultHeaders: {
      "api-key": process.env.AZURE_OPENAI_API_KEY,
    },
  });

  return openai;
};
