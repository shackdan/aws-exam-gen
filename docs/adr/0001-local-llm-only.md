# Generation and review run on a local LLM only

The Generator and the AI Reviewer both run on a local model (Ollama, or MLX-LM on Apple Silicon), never a hosted API. That keeps the cost per Question at zero while building large Question Banks on hardware we already own: a GTX 1070 with 8 GB of VRAM, or an M1 MacBook. In exchange, Questions come out lower quality than a frontier model would produce, and the small context window limits how much Grounding a prompt can carry. Revisit this if Review quality, rather than volume, becomes the thing holding progress back.

## Consequences

- The same model currently plays both the Generator and AI Reviewer roles. A model grading its own output tends to be lenient, which weakens Review. This is a known weakness, not an intended design.
