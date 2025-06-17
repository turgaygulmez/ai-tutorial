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
                link: "/sections/core/machine-learning/machine-learning",
              },
              {
                text: "Open AI",
                link: "/sections/core/open-ai/open-ai",
              },
              {
                text: "LangChain",
                link: "/sections/core/lang-chain/lang-chain",
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
