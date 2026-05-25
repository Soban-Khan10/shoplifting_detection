# Person Matching Demo Plan

This feature is a simulated identity matching demo only.

- It does not perform real face recognition.
- It does not use real personal data.
- It uses a fake database of fictional profiles and a deterministic filename hash to pick a match.
- It is useful for dashboard wiring, UI placeholders, and end-to-end demo flows.

The simulated flow is:

```text
suspicious key frame -> simple center crop -> fake profile lookup -> demo confidence -> JSON result
```

A future real implementation would require face detection, tracking, embedding extraction, database search, and explicit privacy/ethics controls. That is intentionally out of scope here.
