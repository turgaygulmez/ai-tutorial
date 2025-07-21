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
            ],
          },
          {
            text: "Libraries",
            items: [
              {
                text: "LangChain",
                link: "/sections/libraries/lang-chain",
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
