import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";

const port = Number(globalThis.process.env.WEBLLM_BROWSER_TEST_PORT ?? 4178);
const harnessPath = fileURLToPath(
  new globalThis.URL("../../.browser-test/harness.js", import.meta.url),
);

createServer((request, response) => {
  if (request.url?.startsWith("/harness.js")) {
    response.writeHead(200, { "content-type": "text/javascript" });
    createReadStream(harnessPath).pipe(response);
    return;
  }
  response.writeHead(200, { "content-type": "text/html" });
  response.end(
    '<!doctype html><meta charset="utf-8"><script type="module" src="/harness.js"></script>',
  );
}).listen(port, "127.0.0.1");
