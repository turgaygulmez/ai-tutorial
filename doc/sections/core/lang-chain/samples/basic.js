import { getChatModel } from "./core.js";

async function main() {
  const chatModel = getChatModel();
  const result = await chatModel.invoke("Who was the first president of USA?");
  console.log(result.text);
}

main();
