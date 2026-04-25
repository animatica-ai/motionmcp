import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// Env-driven so the same docs source can serve from multiple hosts:
//   - default (`npm start`, mmcp.dev) → "/"
//   - mounted under animatica.ai/mmcp/ → DOCUSAURUS_BASE_URL=/mmcp/ DOCUSAURUS_URL=https://animatica.ai
const baseUrl = process.env.DOCUSAURUS_BASE_URL || "/";
const url     = process.env.DOCUSAURUS_URL     || "https://mmcp.dev";

const config: Config = {
  title: "MMCP",
  tagline: "A simple HTTP contract for motion generation models — and a Python SDK to ship one.",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  url,
  baseUrl,

  organizationName: "animatica-ai",
  projectName: "motionmcp",

  onBrokenLinks: "warn",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          routeBasePath: "docs",
          editUrl: undefined,
          showLastUpdateTime: false,
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/social-card.png",
    metadata: [
      {
        name: "description",
        content:
          "MMCP — Motion Model Context Protocol. A vendor-neutral HTTP contract for text- and constraint-driven character motion generation.",
      },
      { name: "theme-color", content: "#1E1E1E" },
    ],
    colorMode: {
      defaultMode: "dark",
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: "MMCP",
      logo: {
        alt: "MMCP",
        src: "img/logo.svg",
        width: 28,
        height: 28,
      },
      hideOnScroll: false,
      items: [
        {
          type: "docSidebar",
          sidebarId: "getStartedSidebar",
          position: "left",
          label: "Get started",
        },
        {
          type: "docSidebar",
          sidebarId: "sdkSidebar",
          position: "left",
          label: "SDK",
        },
        {
          type: "docSidebar",
          sidebarId: "referenceSidebar",
          position: "left",
          label: "Protocol",
        },
        {
          href: "https://pypi.org/project/motionmcp/",
          position: "right",
          label: "PyPI",
        },
        {
          href: "https://github.com/animatica-ai/motionmcp",
          position: "right",
          className: "header-github-link",
          "aria-label": "GitHub",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Get started",
          items: [
            { label: "Introduction", to: "/docs/get-started/introduction" },
            { label: "Quickstart", to: "/docs/get-started/quickstart" },
          ],
        },
        {
          title: "SDK",
          items: [
            { label: "Install", to: "/docs/sdk/install" },
            { label: "Backbone", to: "/docs/sdk/backbone" },
            { label: "Errors", to: "/docs/sdk/errors" },
            { label: "Serving", to: "/docs/sdk/serving" },
          ],
        },
        {
          title: "Protocol",
          items: [
            { label: "Concepts", to: "/docs/concepts/skeleton" },
            { label: "Reference", to: "/docs/reference/endpoints" },
            { label: "Versioning", to: "/docs/protocol/versioning" },
            { label: "Roadmap", to: "/docs/protocol/roadmap" },
          ],
        },
        {
          title: "Project",
          items: [
            { label: "PyPI", href: "https://pypi.org/project/motionmcp/" },
            { label: "GitHub", href: "https://github.com/animatica-ai/motionmcp" },
          ],
        },
      ],
      copyright: `MMCP — vendor-neutral motion protocol. © ${new Date().getFullYear()}.`,
    },
    prism: {
      theme: prismThemes.vsDark,
      darkTheme: prismThemes.vsDark,
      additionalLanguages: ["bash", "json", "python"],
    },
    docs: {
      sidebar: {
        hideable: false,
        autoCollapseCategories: false,
      },
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
