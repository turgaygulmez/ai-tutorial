import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "AI Tutorial",
  description: "AI Tutorial samples",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      { text: "Documentation", link: "/introduction" },
    ],

    sidebar: [
      {
        text: "Documentation",
        items: [
          {
            text: "Introduction",
            link: "/introduction",
          },
          {
            text: "Core",
            items: [
              {
                text: "Machine Learning",
                link: "/sections/core/base",
              },
              {
                text: "CheatSheet",
                link: "/sections/core/cheat-sheet",
              },
            ],
          },
          {
            text: "Models",
            items: [
              {
                text: "Open AI",
                link: "/sections/models/open-ai",
              },
              {
                text: "Anthropic",
                link: "/sections/models/anthropic",
              },
              {
                text: "Hugging Face",
                link: "/sections/models/huggingface",
              },
            ],
          },
          {
            text: "Libraries",
            items: [
              {
                text: "LangChain",
                link: "/sections/libraries/lang-chain",
              },
              {
                text: "TensorFlow",
                link: "/sections/libraries/tensorflow",
              },
              {
                text: "LlamaNode",
                link: "/sections/libraries/llama-node",
              },
              {
                text: "ThinkChain",
                link: "/sections/libraries/thinkchain",
              },
            ],
          },
          {
            text: "Tools",
            items: [
              {
                text: "Claude Code",
                link: "/sections/tools/claude-code",
              },
              {
                text: "Running LLMs Locally",
                link: "/sections/tools/llm-local",
              },
              {
                text: "PIXIU - Financial AI",
                link: "/sections/tools/pixiu",
              },
              {
                text: "FinGPT - Financial LLM",
                link: "/sections/tools/fingpt",
              },
              {
                text: "MetaGPT - Multi-Agent Framework",
                link: "/sections/tools/metagpt",
              },
              {
                text: "Cheshire Cat AI",
                link: "/sections/tools/cheshire-cat",
              },
              {
                text: "LLaVA - Vision Language Model",
                link: "/sections/tools/llava",
              },
              {
                text: "Onyx - Enterprise AI Assistant",
                link: "/sections/tools/onyx",
              },
              {
                text: "DocsGPT - Documentation Assistant",
                link: "/sections/tools/docsgpt",
              },
              {
                text: "AI Hedge Fund - Intelligent Trading System",
                link: "/sections/tools/ai-hedge-fund",
              },
              {
                text: "Model Context Protocol",
                link: "/sections/tools/mcp",
              },
            ],
          },
        ],
      },
    ],

    socialLinks: [
      { icon: "github", link: "https://github.com/vuejs/vitepress" },
    ],
  },
});
