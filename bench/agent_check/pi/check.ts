// The agent check's pi extension (loaded with -e). It fixes the sampling of every request, so all arms and servers
// sample alike whatever their command-line defaults (the makers' settings: temperature 1.0, top-p 0.95, top-k 20,
// min-p 0), caps each reply at max_tokens, and tags each request for the logging proxy with an X-Agent-Check header
// "<arm>/<task>/<attempt>/<try>/<turn>". AC_TAG, AC_SAMPLING (JSON) and AC_MAX_TOKENS come from run.py.
export default function (pi) {
  const tag = process.env.AC_TAG ?? "untagged";
  const sampling = JSON.parse(process.env.AC_SAMPLING ?? '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0}');
  const maxTokens = Number(process.env.AC_MAX_TOKENS ?? 32768);
  let turn = 0;
  pi.on("turn_start", () => {
    turn += 1;
  });
  pi.on("before_provider_headers", (event) => {
    event.headers["X-Agent-Check"] = `${tag}/${turn}`;
  });
  pi.on("before_provider_request", (event) => {
    const p = { ...event.payload };
    for (const f of ["max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "seed"]) delete p[f];
    return { ...p, ...sampling, max_tokens: maxTokens };
  });
}
