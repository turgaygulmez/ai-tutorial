import { getOpenAI } from "./core.js";

async function main() {
  const openai = getOpenAI();

  const chatCompletion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Who was the first president of USA?" },
    ],
  });

  console.log(chatCompletion?.choices?.[0]?.message?.content);
}

main();
