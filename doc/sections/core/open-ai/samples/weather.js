import { getOpenAI } from "./core.js";

async function main() {
  const openai = getOpenAI();

  const chatCompletion = await openai.chat.completions.create({
    messages: [
      { role: "user", content: "What was the weather on 16/01/2024?" },
    ],
  });

  console.log(chatCompletion?.choices?.[0]?.message?.content);
}

main();
