import OpenAI from "openai";
import { config } from "dotenv";

config();

const openai = new OpenAI({
  apiKey: process.env["OPENAI_API_KEY"], // This is the default and can be omitted
});

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    messages: [
      { role: "user", content: "What was the weather on 16/01/2024?" },
    ],
    model: "gpt-4-1106-preview",
  });

  console.log(chatCompletion?.choices?.[0]?.message?.content);
}

main();
