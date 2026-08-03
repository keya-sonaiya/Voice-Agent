# Decisions

- The supplied state contract has no caller-identity field, so account/order mock
  tools require an identity value supplied and verified server-side by their caller;
  they are intentionally not invoked from the RAG prompt.
- The prompt specifies no external human-queue URL or service credential. The
  `/internal/handoff/{session_id}` adapter is therefore authenticated with the
  configured service secret and returns a payload reconstructed solely from persisted
  state. A deployment can connect that protected adapter to its queue.
- Retrieval uses hybrid dense and BM25 ranking. Run `python -m app.rag.ingest` before
  serving production traffic to create the persistent Chroma index; until then the
  retriever uses the configured sentence-transformer against the bundled corpus.
- TTS output is emitted in chunks over the existing duplex WebSocket. Coqui synthesis
  itself produces a WAV buffer before those chunks; provider-native incremental audio
  streaming is the remaining production integration point.
- A fixed, equal-weight rolling sentiment update is used because the specification
  defines the score but not its smoothing coefficient. It is deterministic and does
  not introduce an additional threshold or configuration value.
- Calls exceeding the configured maximum duration are terminated, the explicit
  termination option permitted by the specification, because no escalation reason for
  duration expiry exists in the required `EscalationDecision` literal union.
- Next.js is pinned to 16.2.12 rather than the supplied 15.1.3 because the latter has
  published critical and high-severity advisories. This exact, React-19-compatible
  version removes those production dependency findings.
- The package manager applies exact `postcss` and `sharp` overrides (8.5.25 and
  0.35.3) to remove the remaining audited transitive findings in Next's dependency
  tree. They are not new runtime features; they are security-only dependency patches.
- `httpx` is pinned to 0.27.2 rather than the supplied 0.28.1 because the supplied
  `ollama==0.4.4` client requires `httpx<0.28.0`, otherwise the exact manifests
  cannot be installed together.
- Windows uses `webrtcvad-wheels==2.0.14` in place of the source-only
  `webrtcvad==2.0.10`. It exposes the same `webrtcvad` import used by the app while
  avoiding a local C++ compilation for that package.
