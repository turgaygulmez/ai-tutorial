import { getOpenAI } from "./core.js";

async function main() {
  const openai = getOpenAI();

  const chatCompletion = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content: `You are a helpful assistant that reports weather. 
          Below are being provided weather data together with their respective date.
          16/01/2024 7°C, 15/01/2024 12°C, 14/01/2024 4°C, 13/01/2024 22°C
          `,
      },
      { role: "user", content: "What was the weather on 16/01/2024?" },
    ],
  });

  console.log(chatCompletion?.choices?.[0]?.message?.content);
}

main();
