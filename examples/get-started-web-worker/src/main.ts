import * as webllm from "@mlc-ai/web-llm";

function markUIFlushStart(tag: string) {
  const mark = `webllm.ui.flush.start:${tag}`;
  performance.mark(mark);
  (console as any).timeStamp?.(mark);
}

function markUIFlushEnd(tag: string) {
  const start = `webllm.ui.flush.start:${tag}`;
  const end = `webllm.ui.flush.end:${tag}`;
  performance.mark(end);
  performance.measure(`webllm.ui.flush:${tag}`, { start, end });
  performance.clearMarks(start);
  performance.clearMarks(end);
  performance.clearMeasures(`webllm.ui.flush:${tag}`);
  (console as any).timeStamp?.(end);
}

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

// There are two demonstrations, pick one to run

/**
 * Chat completion (OpenAI style) without streaming, where we get the entire response at once.
 */
async function mainNonStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";

  const engine: webllm.MLCEngineInterface =
    await webllm.CreateWebWorkerMLCEngine(
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
      selectedModel,
      { initProgressCallback: initProgressCallback },
    );

  const request: webllm.ChatCompletionRequest = {
    messages: [
      {
        role: "system",
        content:
          "You are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please. ",
      },
      { role: "user", content: "Provide me three US states." },
      { role: "assistant", content: "California, New York, Pennsylvania." },
      { role: "user", content: "Two more please!" },
    ],
    n: 3,
    temperature: 1.5,
    max_tokens: 256,
    extra_body: {
      enable_trace: true,
      trace_level: "major",
      trace_devtools: "major",
      enable_gpu_timestamps: true,
    },
  };

  const reply0 = await engine.chat.completions.create(request);
  markUIFlushStart("web-worker.reply");
  setLabel("generate-label", reply0.choices[0]?.message.content || "");
  markUIFlushEnd("web-worker.reply");
  console.log(reply0);

  console.log(reply0.usage);
  console.log(await engine.drainTraceEvents({ clear: true }));
}

/**
 * Chat completion (OpenAI style) with streaming, where delta is sent while generating response.
 */
async function mainStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";

  const engine: webllm.MLCEngineInterface =
    await webllm.CreateWebWorkerMLCEngine(
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
      selectedModel,
      { initProgressCallback: initProgressCallback },
    );

  const request: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      {
        role: "system",
        content:
          "You are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please. ",
      },
      { role: "user", content: "Provide me three US states." },
      { role: "assistant", content: "California, New York, Pennsylvania." },
      { role: "user", content: "Two more please!" },
    ],
    temperature: 1.5,
    max_tokens: 256,
    extra_body: {
      enable_trace: true,
      trace_level: "major",
      trace_devtools: "major",
      enable_gpu_timestamps: true,
    },
  };

  const asyncChunkGenerator = await engine.chat.completions.create(request);
  let message = "";
  for await (const chunk of asyncChunkGenerator) {
    console.log(chunk);
    message += chunk.choices[0]?.delta?.content || "";
    markUIFlushStart("web-worker.stream");
    setLabel("generate-label", message);
    markUIFlushEnd("web-worker.stream");
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }
  console.log("Final message:\n", await engine.getMessage()); // the concatenated message
  console.log(await engine.drainTraceEvents({ clear: true }));
}

// Run one of the function below
// mainNonStreaming();
mainStreaming();
