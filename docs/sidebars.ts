import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  getStartedSidebar: [
    {
      type: "category",
      label: "Get started",
      collapsed: false,
      collapsible: false,
      items: [
        "get-started/introduction",
        "get-started/quickstart",
        "get-started/quickstart-client",
        "get-started/implementations",
      ],
    },
  ],

  sdkSidebar: [
    {
      type: "category",
      label: "Python SDK",
      collapsed: false,
      collapsible: false,
      items: [
        "sdk/index",
        "sdk/install",
        "sdk/backbone",
        "sdk/results",
        "sdk/errors",
        "sdk/serving",
        "sdk/null-backbone",
      ],
    },
  ],

  referenceSidebar: [
    {
      type: "category",
      label: "Concepts",
      collapsed: false,
      collapsible: false,
      items: [
        "concepts/skeleton",
        "concepts/segments",
        "concepts/constraints",
        "concepts/timing",
        "concepts/coordinates",
      ],
    },
    {
      type: "category",
      label: "Reference",
      collapsed: false,
      collapsible: false,
      items: [
        "reference/endpoints",
        "reference/capabilities",
        "reference/generate-request",
        "reference/response",
        "reference/errors",
        "reference/async",
      ],
    },
    {
      type: "category",
      label: "Protocol",
      collapsed: false,
      collapsible: false,
      items: ["protocol/versioning", "protocol/roadmap"],
    },
  ],
};

export default sidebars;
