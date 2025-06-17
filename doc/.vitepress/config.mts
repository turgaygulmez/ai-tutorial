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
                link: "/sections/core/machine-learning",
              },
              {
                text: "Open AI",
                link: "/sections/core/open-ai",
              },
              {
                text: "Claude",
                link: "/sections/core/claude",
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
                text: "Claude Code",
                link: "/sections/libraries/claude-code",
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
