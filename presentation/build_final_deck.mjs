import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const artifactModule =
  "file:///C:/Users/tahar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/@oai+artifact-tool@file+loc_40138bddc7e6fa8f5488e20ef9bf79a4/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  shape,
  text,
  image,
  chart,
  rule,
  fill,
  fixed,
  hug,
  wrap,
  grow,
  fr,
  auto,
} = await import(artifactModule);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const outDir = path.join(__dirname, "output");
const previewDir = path.join(__dirname, "scratch", "previews");
const qaDir = path.join(__dirname, "scratch", "qa");
fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(previewDir, { recursive: true });
fs.mkdirSync(qaDir, { recursive: true });
for (const dir of [previewDir, qaDir]) {
  for (const file of fs.readdirSync(dir)) {
    fs.rmSync(path.join(dir, file), { force: true });
  }
}

async function saveExportedAsset(asset, outPath) {
  if (asset && typeof asset.save === "function") {
    await asset.save(outPath);
    return;
  }
  if (asset && typeof asset.arrayBuffer === "function") {
    fs.writeFileSync(outPath, Buffer.from(await asset.arrayBuffer()));
    return;
  }
  if (typeof asset === "string") {
    const match = asset.match(/^data:.*?;base64,(.*)$/);
    fs.writeFileSync(outPath, match ? Buffer.from(match[1], "base64") : asset);
    return;
  }
  if (asset instanceof Uint8Array) {
    fs.writeFileSync(outPath, Buffer.from(asset));
    return;
  }
  throw new Error(`Unsupported export payload for ${outPath}: ${Object.prototype.toString.call(asset)}`);
}

const W = 1920;
const H = 1080;
const C = {
  ink: "#111827",
  muted: "#536171",
  faint: "#D6DEE7",
  paper: "#F8F4EA",
  white: "#FFFFFF",
  green: "#0B6B5A",
  mint: "#B8E3D3",
  blue: "#2357A6",
  sky: "#BBD8FF",
  amber: "#E7A538",
  coral: "#D9654F",
  night: "#102033",
};

const T = {
  display: "Georgia",
  body: "Aptos",
  mono: "Consolas",
};

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

function addSlide(name, node) {
  const slide = presentation.slides.add();
  slide.name = name;
  slide.compose(node, {
    frame: { left: 0, top: 0, width: W, height: H },
    baseUnit: 8,
  });
  return slide;
}

function bg(color = C.paper) {
  return shape({ name: "background", width: fill, height: fill, fill: color, line: { color, transparency: 100 } });
}

function pngDataUrl(filePath) {
  return `data:image/png;base64,${fs.readFileSync(filePath).toString("base64")}`;
}

function footer(label) {
  return row(
    { name: "footer", width: fill, height: hug, align: "center", justify: "between" },
    [
      text(label, {
        name: "footer-label",
        width: wrap(900),
        height: hug,
        style: { fontFace: T.body, fontSize: 17, color: C.muted },
      }),
      text("Drift-Aware MLOps Pipeline | ACS Income Data", {
        name: "footer-context",
        width: wrap(720),
        height: hug,
        style: { fontFace: T.body, fontSize: 17, color: C.muted },
      }),
    ],
  );
}

function titleStack(kicker, title, subtitle, color = C.ink) {
  return column(
    { name: "title-stack", width: fill, height: hug, gap: 12 },
    [
      text(kicker.toUpperCase(), {
        name: "kicker",
        width: fill,
        height: hug,
        style: { fontFace: T.mono, fontSize: 18, color: C.green, bold: true, characterSpacing: 2.5 },
      }),
      text(title, {
        name: "slide-title",
        width: wrap(1300),
        height: hug,
        style: { fontFace: T.display, fontSize: 56, color, bold: false },
      }),
      text(subtitle, {
        name: "slide-subtitle",
        width: wrap(1180),
        height: hug,
        style: { fontFace: T.body, fontSize: 25, color: C.muted },
      }),
    ],
  );
}

function metric(label, value, note, color = C.green) {
  return column(
    { name: `metric-${label}`, width: fill, height: hug, gap: 6 },
    [
      text(value, {
        name: `metric-value-${label}`,
        width: fill,
        height: hug,
        style: { fontFace: T.display, fontSize: 66, color, bold: true },
      }),
      text(label.toUpperCase(), {
        name: `metric-label-${label}`,
        width: fill,
        height: hug,
        style: { fontFace: T.mono, fontSize: 15, color: C.ink, bold: true, characterSpacing: 1.8 },
      }),
      text(note, {
        name: `metric-note-${label}`,
        width: wrap(360),
        height: hug,
        style: { fontFace: T.body, fontSize: 19, color: C.muted },
      }),
    ],
  );
}

function pill(textValue, color = C.green, bgColor = "#E7F4EF") {
  return row(
    {
      name: `pill-${textValue}`,
      width: hug,
      height: fixed(42),
      align: "center",
      padding: { x: 18, y: 6 },
      gap: 8,
    },
    [
      shape({ width: fixed(12), height: fixed(12), fill: color, borderRadius: "rounded-full", line: { color, transparency: 100 } }),
      text(textValue, {
        width: hug,
        height: hug,
        style: { fontFace: T.body, fontSize: 19, color, bold: true },
      }),
    ],
  );
}

function thinBox(name, children, accent = C.green) {
  return column(
    {
      name,
      width: fill,
      height: fill,
      gap: 12,
      padding: { x: 26, y: 22 },
    },
    [
      shape({ name: `${name}-rule`, width: fixed(110), height: fixed(6), fill: accent, line: { color: accent, transparency: 100 } }),
      ...children,
    ],
  );
}

function miniStep(num, heading, body, color) {
  return row(
    { name: `step-${num}`, width: fill, height: hug, gap: 20, align: "start" },
    [
      shape({ name: `step-dot-${num}`, width: fixed(46), height: fixed(46), fill: color, borderRadius: "rounded-full", line: { color, transparency: 100 } }),
      column(
        { width: fill, height: hug, gap: 4 },
        [
          text(`${num}. ${heading}`, {
            name: `step-heading-${num}`,
            width: fill,
            height: hug,
            style: { fontFace: T.body, fontSize: 25, color: C.ink, bold: true },
          }),
          text(body, {
            name: `step-body-${num}`,
            width: wrap(560),
            height: hug,
            style: { fontFace: T.body, fontSize: 20, color: C.muted },
          }),
        ],
      ),
    ],
  );
}

addSlide(
  "cover",
  layers(
    { width: fill, height: fill },
    [
      bg(C.paper),
      grid(
        {
          name: "cover-grid",
          width: fill,
          height: fill,
          columns: [fr(1.03), fr(0.97)],
          padding: { x: 86, y: 70 },
          columnGap: 56,
        },
        [
          column(
            { name: "cover-copy", width: fill, height: fill, justify: "between" },
            [
              column(
                { width: fill, height: hug, gap: 22 },
                [
                  text("WHEN DOES ADAPTIVE RETRAINING HELP?", {
                    name: "cover-kicker",
                    width: fill,
                    height: hug,
                    style: { fontFace: T.mono, fontSize: 19, color: C.green, bold: true, characterSpacing: 2.1 },
                  }),
                  text("A Drift-Aware MLOps Study on Multi-Year ACS Income Data", {
                    name: "cover-title",
                    width: wrap(880),
                    height: hug,
                    style: { fontFace: T.display, fontSize: 67, color: C.ink, bold: false },
                  }),
                  text("End-to-end ML pipeline with experiment tracking, CI/CD, containerized serving, monitoring, and adaptive retraining under dataset shift.", {
                    name: "cover-subtitle",
                    width: wrap(800),
                    height: hug,
                    style: { fontFace: T.body, fontSize: 26, color: C.muted },
                  }),
                ],
              ),
              column(
                { width: fill, height: hug, gap: 10 },
                [
                  text("Taha Rasheed | Hamza Zahid | Hasnat Noor", {
                    name: "authors",
                    width: fill,
                    height: hug,
                    style: { fontFace: T.body, fontSize: 25, color: C.ink, bold: true },
                  }),
                  text("National University of Computers and Emerging Sciences, ISB", {
                    name: "affiliation",
                    width: fill,
                    height: hug,
                    style: { fontFace: T.body, fontSize: 20, color: C.muted },
                  }),
                ],
              ),
            ],
          ),
          layers(
            { name: "cover-art", width: fill, height: fill },
            [
              shape({ name: "cover-plane", width: fill, height: fill, fill: C.night, line: { color: C.night, transparency: 100 } }),
              grid(
                { width: fill, height: fill, columns: [fr(1)], rows: [fr(1), fr(1), fr(1), fr(1)], padding: { x: 70, y: 76 }, rowGap: 22 },
                [
                  row({ width: fill, height: fill, align: "center", gap: 18 }, [
                    text("STATIC", { width: fixed(142), height: hug, style: { fontFace: T.mono, fontSize: 20, color: C.sky, bold: true } }),
                    shape({ width: fixed(300), height: fixed(8), fill: "#4E6C8F", line: { color: "#4E6C8F", transparency: 100 } }),
                    text("0.710", { width: fixed(110), height: hug, style: { fontFace: T.mono, fontSize: 24, color: C.sky, bold: true } }),
                  ]),
                  row({ width: fill, height: fill, align: "center", gap: 18 }, [
                    text("POLICY", { width: fixed(142), height: hug, style: { fontFace: T.mono, fontSize: 20, color: C.mint, bold: true } }),
                    shape({ width: fixed(370), height: fixed(8), fill: C.mint, line: { color: C.mint, transparency: 100 } }),
                    text("0.734", { width: fixed(110), height: hug, style: { fontFace: T.mono, fontSize: 24, color: C.mint, bold: true } }),
                  ]),
                  row({ width: fill, height: fill, align: "center", gap: 18 }, [
                    text("DRIFT", { width: fixed(142), height: hug, style: { fontFace: T.mono, fontSize: 20, color: C.amber, bold: true } }),
                    shape({ width: fixed(260), height: fixed(8), fill: C.amber, line: { color: C.amber, transparency: 100 } }),
                    text("12 batches", { width: fixed(190), height: hug, style: { fontFace: T.mono, fontSize: 24, color: C.amber, bold: true } }),
                  ]),
                  text("The core result: retraining helps when the new regime is learnable, not when drift is synthetic label noise.", {
                    name: "cover-art-caption",
                    width: wrap(720),
                    height: hug,
                    style: { fontFace: T.display, fontSize: 36, color: C.white },
                  }),
                ],
              ),
            ],
          ),
        ],
      ),
    ],
  ),
);

addSlide(
  "problem",
  layers(
    { width: fill, height: fill },
    [
      bg(C.white),
      column(
        { width: fill, height: fill, padding: { x: 86, y: 60 }, gap: 42 },
        [
          titleStack(
            "Research problem",
            "Static models decay quietly when the data-generating world moves",
            "We studied whether monitored, policy-gated retraining can recover performance without retraining on every alert.",
          ),
          grid(
            { width: fill, height: fill, columns: [fr(1.12), fr(0.88)], columnGap: 58 },
            [
              column(
                { width: fill, height: fill, gap: 28 },
                [
                  miniStep("A", "Dataset shift", "The input distribution changes across ACS years and simulated batch regimes.", C.blue),
                  miniStep("B", "Concept signal", "Labels and performance are monitored to distinguish harmless drift from harmful degradation.", C.amber),
                  miniStep("C", "Retraining decision", "Policy decides whether new labeled batches justify replacing the deployed model.", C.green),
                ],
              ),
              column(
                { width: fill, height: fill, gap: 24, justify: "center" },
                [
                  metric("Static batch-12 F1", "0.710", "Final batch performance after sustained drift.", C.coral),
                  rule({ width: fixed(520), stroke: C.faint, weight: 2 }),
                  metric("Policy batch-12 F1", "0.734", "Recovered by adapting to recent batches.", C.green),
                ],
              ),
            ],
          ),
          footer("Problem framing and final ACS batch signal"),
        ],
      ),
    ],
  ),
);

addSlide(
  "system",
  layers(
    { width: fill, height: fill },
    [
      bg(C.paper),
      column(
        { width: fill, height: fill, padding: { x: 82, y: 56 }, gap: 34 },
        [
          titleStack(
            "End-to-end implementation",
            "Full MLOps loop",
            "Not only a model: data processing, XGBoost training, drift detection, policy retraining, serving, tracking, monitoring, and CI/CD are integrated.",
          ),
          grid(
            { width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 24, rowGap: 22 },
            [
              thinBox("data-box", [
                text("Data + Features", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 28, bold: true, color: C.ink } }),
                text("Adult + multi-year ACS/Folktables; temporal split; 4.8M processed ACS rows.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 22, color: C.muted } }),
              ], C.blue),
              thinBox("train-box", [
                text("Training", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 28, bold: true, color: C.ink } }),
                text("XGBoost with CPU/GPU support; encoders and model artifacts saved for serving.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 22, color: C.muted } }),
              ], C.green),
              thinBox("drift-box", [
                text("Drift + Policy", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 28, bold: true, color: C.ink } }),
                text("KS / chi-square tests, severity scoring, concept-drift gates, rolling retraining windows.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 22, color: C.muted } }),
              ], C.amber),
              thinBox("serve-box", [
                text("FastAPI", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 28, bold: true, color: C.ink } }),
                text("Prediction API, health endpoint, model reload, and Prometheus `/metrics`.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 22, color: C.muted } }),
              ], C.coral),
              thinBox("track-box", [
                text("MLflow", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 28, bold: true, color: C.ink } }),
                text("Experiment parameters, metrics, model versions, and training artifacts.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 22, color: C.muted } }),
              ], C.blue),
              thinBox("ops-box", [
                text("Docker + CI/CD + Grafana", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 28, bold: true, color: C.ink } }),
                text("Compose stack, Prometheus scrape, Grafana dashboard, GitHub Actions validation.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 22, color: C.muted } }),
              ], C.green),
            ],
          ),
          footer("Implementation stack mapped to mandatory MLOps requirements"),
        ],
      ),
    ],
  ),
);

addSlide(
  "method",
  layers(
    { width: fill, height: fill },
    [
      bg(C.white),
      column(
        { width: fill, height: fill, padding: { x: 86, y: 60 }, gap: 38 },
        [
          titleStack(
            "Methodology upgrade",
            "The important change: stop asking the model to learn noise",
            "Initial label-flip drift created contradictions; final ACS experiments focus on learnable temporal, covariate, and feature-regime shifts.",
          ),
          grid(
            { width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 48 },
            [
              column(
                { width: fill, height: fill, gap: 22 },
                [
                  text("Before", { width: fill, height: hug, style: { fontFace: T.display, fontSize: 44, color: C.coral, bold: true } }),
                  miniStep("1", "Subgroup label flips", "Some income labels were artificially reversed for selected groups.", C.coral),
                  miniStep("2", "Short drift spikes", "Retraining often adapted to a batch that the next batch did not resemble.", C.coral),
                  miniStep("3", "Low recovery signal", "Static and retrained models stayed close because both faced noisy or unstable targets.", C.coral),
                ],
              ),
              column(
                { width: fill, height: fill, gap: 22 },
                [
                  text("After", { width: fill, height: hug, style: { fontFace: T.display, fontSize: 44, color: C.green, bold: true } }),
                  miniStep("1", "Learnable drift regimes", "Age, hours, occupation, birthplace, and feature-noise shifts across 12 batches.", C.green),
                  miniStep("2", "Rolling retraining", "Train on recent batches plus a small historical anchor, instead of unbounded expansion.", C.green),
                  miniStep("3", "Persistence gating", "Policy waits for repeated drift/concept signals before replacing the model.", C.green),
                ],
              ),
            ],
          ),
          footer("Core methodological correction after noise-drift diagnosis"),
        ],
      ),
    ],
  ),
);

addSlide(
  "results",
  layers(
    { width: fill, height: fill },
    [
      bg(C.paper),
      column(
        { width: fill, height: fill, padding: { x: 82, y: 56 }, gap: 30 },
        [
          titleStack(
            "Main ACS result",
            "Policy retraining recovered late-regime performance with fewer retrains",
            "The adaptive policy closed much of the late-batch degradation while using almost half the retraining events of immediate retraining.",
          ),
          grid(
            { width: fill, height: fill, columns: [fr(1.08), fr(0.92)], columnGap: 42 },
            [
              chart({
                name: "batch12-chart",
                chartType: "bar",
                width: fill,
                height: fill,
                config: {
                  title: "Batch 12 F1 and Accuracy",
                  categories: ["Static", "Immediate", "Policy"],
                  series: [
                    { name: "F1", values: [0.710, 0.732, 0.734] },
                    { name: "Accuracy", values: [0.730, 0.778, 0.783] },
                  ],
                },
              }),
              column(
                { width: fill, height: fill, gap: 24, justify: "center" },
                [
                  metric("Static final accuracy", "0.730", "Baseline remains locked to the original regime.", C.coral),
                  metric("Policy final accuracy", "0.783", "Adaptive model recovers under sustained drift.", C.green),
                  metric("Retrains avoided", "5 fewer", "Policy retrained 6 times vs 11 immediate retrains.", C.blue),
                ],
              ),
            ],
          ),
          footer("Source: reports/results_acs.md"),
        ],
      ),
    ],
  ),
);

addSlide(
  "ablation",
  layers(
    { width: fill, height: fill },
    [
      bg(C.white),
      column(
        { width: fill, height: fill, padding: { x: 82, y: 56 }, gap: 32 },
        [
          titleStack(
            "Why retraining sometimes failed",
            "Retraining helps only when the new batch contains a learnable regime",
            "Noise ablation separated real shift from subgroup label-flip stress, explaining why the earliest experiment looked flat.",
          ),
          grid(
            { width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 36 },
            [
              image({
                name: "mean-f1-ablation",
                dataUrl: pngDataUrl(path.join(root, "reports", "diagnostics", "acs_noise_ablation", "mean_f1_by_ablation.png")),
                contentType: "image/png",
                width: fill,
                height: fill,
                fit: "contain",
                alt: "Mean F1 by noise ablation.",
              }),
              column(
                { width: fill, height: fill, gap: 28, justify: "center" },
                [
                  text("Interpretation", { width: fill, height: hug, style: { fontFace: T.display, fontSize: 44, color: C.ink } }),
                  miniStep("1", "Label flips are not a stable concept", "They inject contradictions into the target, so a model cannot generalize the new rule.", C.coral),
                  miniStep("2", "Sliding windows matter", "Recent data dominates adaptation; old data should not drown out the current regime.", C.blue),
                  miniStep("3", "The final result is conditional", "Adaptive retraining is valuable when drift is persistent and learnable.", C.green),
                ],
              ),
            ],
          ),
          footer("Noise ablation and regime diagnostics"),
        ],
      ),
    ],
  ),
);

addSlide(
  "ops",
  layers(
    { width: fill, height: fill },
    [
      bg(C.night),
      column(
        { width: fill, height: fill, padding: { x: 82, y: 56 }, gap: 34 },
        [
          column(
            { width: fill, height: hug, gap: 12 },
            [
              text("PRODUCTION READINESS", { width: fill, height: hug, style: { fontFace: T.mono, fontSize: 18, color: C.mint, bold: true, characterSpacing: 2.2 } }),
              text("The pipeline is demonstrable as a running MLOps system", {
                width: wrap(1180),
                height: hug,
                style: { fontFace: T.display, fontSize: 56, color: C.white },
              }),
              text("This is the part to show live: FastAPI, MLflow, Prometheus, Grafana, Docker Compose, and GitHub Actions.", {
                width: wrap(1240),
                height: hug,
                style: { fontFace: T.body, fontSize: 25, color: "#C5D4E3" },
              }),
            ],
          ),
          grid(
            { width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 22, rowGap: 20 },
            [
              thinBox("fastapi", [text("FastAPI", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 30, color: C.white, bold: true } }), text("/predict, /health, /metrics, reload-model.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 21, color: "#C5D4E3" } })], C.mint),
              thinBox("mlflow", [text("MLflow", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 30, color: C.white, bold: true } }), text("Params, metrics, model artifacts, run history.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 21, color: "#C5D4E3" } })], C.sky),
              thinBox("docker", [text("Docker", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 30, color: C.white, bold: true } }), text("API, MLflow, Prometheus, Grafana stack.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 21, color: "#C5D4E3" } })], C.amber),
              thinBox("prom", [text("Prometheus", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 30, color: C.white, bold: true } }), text("Latency, request count, model version, drift metrics.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 21, color: "#C5D4E3" } })], C.coral),
              thinBox("grafana", [text("Grafana", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 30, color: C.white, bold: true } }), text("Dashboard for operational model monitoring.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 21, color: "#C5D4E3" } })], C.mint),
              thinBox("ci", [text("GitHub Actions", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 30, color: C.white, bold: true } }), text("Lint, tests, compile checks, Docker validation/build.", { width: fill, height: hug, style: { fontFace: T.body, fontSize: 21, color: "#C5D4E3" } })], C.sky),
            ],
          ),
          row({ width: fill, height: hug, justify: "between", align: "center" }, [
            text("Live demo URLs: :8080/docs | :5000 MLflow | :9090 Prometheus | :3000 Grafana", {
              width: wrap(1120),
              height: hug,
              style: { fontFace: T.mono, fontSize: 18, color: "#B7C9DD" },
            }),
            text("MLOps stack verification", { width: wrap(520), height: hug, style: { fontFace: T.body, fontSize: 17, color: "#B7C9DD" } }),
          ]),
        ],
      ),
    ],
  ),
);

const pptx = await PresentationFile.exportPptx(presentation);
const pptxPath = path.join(outDir, "Drift_Aware_MLOps_ACS_Presentation.pptx");
await pptx.save(pptxPath);

const previewPaths = [];
for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  const png = await presentation.export({ slide, format: "png" });
  const out = path.join(previewDir, `slide_${String(i + 1).padStart(2, "0")}.png`);
  await saveExportedAsset(png, out);
  previewPaths.push(out);

  const layout = await presentation.export({ slide, format: "layout" });
  fs.writeFileSync(path.join(qaDir, `slide_${String(i + 1).padStart(2, "0")}.layout.json`), JSON.stringify(layout, null, 2));
}

fs.writeFileSync(
  path.join(outDir, "speaker_notes.md"),
  [
    "# Drift-Aware MLOps ACS Presentation - Speaker Notes",
    "",
    "Slide 1: Introduce the project as both a research study and an end-to-end MLOps implementation.",
    "Slide 2: Explain the problem: deployed ML models decay under dataset shift, and retraining must be justified.",
    "Slide 3: Walk through the system components: data, training, drift policy, API, MLflow, Prometheus/Grafana, CI/CD.",
    "Slide 4: Emphasize the methodological correction: label flips created noise; final experiments use learnable drift regimes.",
    "Slide 5: Present the main result: policy retraining improves final batch accuracy/F1 while avoiding unnecessary retrains.",
    "Slide 6: Explain the ablation result: retraining fails under synthetic label noise but helps under learnable shift.",
    "Slide 7: Show the operational stack and close with the thesis: retraining is a signal-quality decision, not a magic button.",
    "",
    "Suggested split: Taha covers slides 1, 2, 5, 7; Hamza covers slide 4 and experiment logic; Hasnat covers slide 3 and slide 6 or the live MLOps demo.",
  ].join("\n"),
);

console.log(JSON.stringify({ pptxPath, previewPaths }, null, 2));
