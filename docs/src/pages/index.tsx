import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import CodeBlock from "@theme/CodeBlock";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./index.module.css";

function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroGrid} aria-hidden="true" />
      <div className={styles.heroInner}>
        <div className={styles.eyebrow}>
          Motion Model Context Protocol — v1.0-rc1
        </div>
        <h1 className={styles.title}>
          One protocol.
          <br />
          <span className={styles.titleAccent}>Any motion model.</span>
        </h1>
        <p className={styles.tagline}>
          A vendor-neutral HTTP contract for text- and constraint-driven
          motion generation. Send a skeleton and a prompt; get back a
          standard glTF&nbsp;2.0 animation. Build motion-capable plugins for
          any DCC without locking yourself to a single AI vendor.
        </p>

        <div className={styles.ctas}>
          <Link to="/docs/get-started/introduction" className={styles.ctaPrimary}>
            Read the protocol <span className={styles.arrow}>→</span>
          </Link>
          <Link
            to="/docs/get-started/quickstart-client"
            className={styles.ctaSecondary}
          >
            Quickstart
          </Link>
        </div>

        <div className={styles.heroMeta}>
          <span>
            <strong>Wire format</strong>
            JSON request, glTF 2.0 response
          </span>
          <span>
            <strong>Python SDK</strong>
            <code>pip install motionmcp</code>
          </span>
          <span>
            <strong>Status</strong>
            Release candidate
          </span>
        </div>
      </div>
    </section>
  );
}

function Verbs() {
  return (
    <section className={styles.section}>
      <div className={styles.sectionInner}>
        <div className={styles.sectionEyebrow}>The contract, in three verbs</div>
        <h2 className={styles.sectionTitle}>Discover. Generate. Decode.</h2>
        <p className={styles.sectionLead}>
          MMCP is two HTTP calls and a glTF document. No SDK to install. No
          vendor lock-in. The same wire format works for every backbone —
          diffusion, transformer, or hybrid.
        </p>

        <div className={styles.verbs}>
          <article className={styles.verb}>
            <span className={styles.verbNumber}>01 — Discover</span>
            <h3 className={styles.verbTitle}>What can the model do?</h3>
            <p className={styles.verbBody}>
              Ask the server which models it hosts, which constraints they
              support, what fps they run at, and what canonical skeleton they
              expect.
            </p>
            <code className={styles.verbCode}>GET /capabilities</code>
          </article>

          <article className={styles.verb}>
            <span className={styles.verbNumber}>02 — Generate</span>
            <h3 className={styles.verbTitle}>Hand it your skeleton.</h3>
            <p className={styles.verbBody}>
              Send the user's actual skeleton — joint names, parent links,
              rest pose. Add a prompt or a few constraints. The server
              returns a glTF document with one animation per sample.
            </p>
            <code className={styles.verbCode}>POST /generate</code>
          </article>

          <article className={styles.verb}>
            <span className={styles.verbNumber}>03 — Decode</span>
            <h3 className={styles.verbTitle}>Apply the animation.</h3>
            <p className={styles.verbBody}>
              The response is standard glTF 2.0 — load it with any glTF
              parser. Optional <code>extensions.MMCP_motion</code> carries
              fps, foot-contact masks, and chunk boundaries.
            </p>
            <code className={styles.verbCode}>model/gltf+json</code>
          </article>
        </div>
      </div>
    </section>
  );
}

function Snippet() {
  return (
    <section className={styles.snippetSection}>
      <div className={styles.snippetGrid}>
        <div className={styles.snippetCopy}>
          <div className={styles.sectionEyebrow}>Hand-writable</div>
          <h2>A minimal request fits in five lines.</h2>
          <p>
            No upload step. No buffers in the request. Just JSON — the
            character's skeleton plus a description of the motion you want.
          </p>
          <p>
            Read the{" "}
            <Link to="/docs/reference/generate-request">request schema</Link>{" "}
            for the full surface, or jump to the{" "}
            <Link to="/docs/get-started/quickstart-client">
              client quickstart
            </Link>{" "}
            to send your first request in under five minutes.
          </p>
        </div>

        <div className={styles.codeCard}>
          <div className={styles.codeHeader}>
            <span className={styles.codeMethod}>POST · /generate</span>
            <span className={styles.codePath}>application/json</span>
          </div>
          <CodeBlock language="json" className={styles.embeddedCode}>
{`{
  "protocol_version": "1.0",
  "model": "kimodo-soma-rp",
  "skeleton": { "joints": [ /* canonical or your own */ ] },
  "segments": [
    {
      "type": "text",
      "prompt": "a person walks forward, then waves",
      "duration_frames": 120
    }
  ]
}`}
          </CodeBlock>
        </div>
      </div>
    </section>
  );
}

function Features() {
  const items = [
    {
      icon: "⊕",
      title: "Vendor-neutral",
      body:
        "Any motion-generation backbone implements the same wire format. Swap models without rewriting your plugin.",
    },
    {
      icon: "{ }",
      title: "Plain JSON in, glTF out",
      body:
        "No proprietary formats. Inputs are a single JSON document; outputs load into any tool that reads glTF 2.0.",
    },
    {
      icon: "⌗",
      title: "Skeleton-respectful",
      body:
        "Send the user's skeleton verbatim. Servers retarget under the hood. No imposed canonical naming.",
    },
    {
      icon: "◐",
      title: "Discoverable",
      body:
        "One GET tells you which models, which constraints, which fps, and what limits. No surprises at runtime.",
    },
    {
      icon: "↻",
      title: "Versioned for the long haul",
      body:
        "Major versions for breaking changes. Minor versions add capabilities. Servers run both concurrently when needed.",
    },
    {
      icon: "✦",
      title: "Hand-writable",
      body:
        "A minimal request is five lines. The contract is small enough to reason about end-to-end in one sitting.",
    },
  ];
  return (
    <section className={styles.featuresSection}>
      <div className={styles.sectionInner}>
        <div className={styles.sectionEyebrow}>Design principles</div>
        <h2 className={styles.sectionTitle}>Built for studios, not demos.</h2>
        <p className={styles.sectionLead}>
          The protocol surface is intentionally small. Implementation details
          — backbones, retargeters, post-processors — sit behind the contract
          where they belong.
        </p>

        <div className={styles.features}>
          {items.map((it) => (
            <article key={it.title} className={styles.feature}>
              <div className={styles.featureIcon}>{it.icon}</div>
              <h3 className={styles.featureTitle}>{it.title}</h3>
              <p className={styles.featureBody}>{it.body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function SdkCallout() {
  return (
    <section className={styles.sdkCallout}>
      <div className={styles.sdkInner}>
        <div className={styles.sdkCopy}>
          <div className={styles.sdkEyebrow}>Building a server in Python?</div>
          <h2 className={styles.sdkTitle}>
            There's an SDK so you don't have to start from scratch.
          </h2>
          <p className={styles.sdkLead}>
            <code>motionmcp</code> is a thin Python package that handles the
            wire format, validation, error envelope, and glTF encoding. You
            write a <code>Backbone</code> subclass; it serves the protocol.
            The SDK is a convenience — the protocol stands alone, in any
            language.
          </p>
          <div className={styles.sdkCtas}>
            <Link to="/docs/sdk/" className={styles.sdkCtaPrimary}>
              SDK reference <span className={styles.arrow}>→</span>
            </Link>
            <Link to="/docs/get-started/quickstart" className={styles.sdkCtaSecondary}>
              Server quickstart
            </Link>
          </div>
          <code className={styles.sdkInstall}>pip install motionmcp</code>
        </div>

        <div className={styles.sdkCodeCard}>
          <div className={styles.codeHeader}>
            <span className={styles.codeMethod}>python · server.py</span>
            <span className={styles.codePath}>motionmcp 0.1</span>
          </div>
          <CodeBlock language="python" className={styles.embeddedCode}>
{`from motionmcp import (
    Backbone, ModelSpec, GenerateRequest,
    MotionResult, serve,
)

class MyBackbone(Backbone):
    def capabilities(self) -> ModelSpec:
        return ModelSpec(
            id="my-model",
            fps=30.0,
            canonical_skeleton=load_skeleton(),
            supported_constraints=["pose_keyframe"],
        )

    async def generate(self, req: GenerateRequest):
        return MotionResult(
            rotations=self.model.run(req),
            root_translations=self.model.run_root(req),
        )

if __name__ == "__main__":
    serve(MyBackbone())`}
          </CodeBlock>
        </div>
      </div>
    </section>
  );
}

function EndCta() {
  return (
    <section className={styles.endCta}>
      <div className={styles.endCtaInner}>
        <h2 className={styles.endCtaTitle}>
          Read the protocol. Send a request. Ship.
        </h2>
        <p className={styles.endCtaLead}>
          The protocol surface is small enough to read end-to-end in one
          sitting. The quickstart sends a real request in under five
          minutes.
        </p>
        <div className={styles.endCtaButtons}>
          <Link to="/docs/get-started/introduction" className={styles.ctaPrimary}>
            Read the protocol <span className={styles.arrow}>→</span>
          </Link>
          <Link to="/docs/concepts/skeleton" className={styles.ctaSecondary}>
            Learn the concepts
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.ReactElement {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="MMCP — Motion Model Context Protocol"
      description={siteConfig.tagline}
    >
      <main className={styles.page}>
        <Hero />
        <Verbs />
        <Snippet />
        <Features />
        <SdkCallout />
        <EndCta />
      </main>
    </Layout>
  );
}
