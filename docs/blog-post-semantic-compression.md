# Four Developers Walk Into a Prompt

*On naming people as a form of semantic compression, and why "Samuel Colvin" is worth 10,000 words of instructions.*

---

I was writing a code review prompt the other day. Not a checklist — I'd tried checklists. "Check for type safety. Check for performance. Check for readability." They produce reviews that read like a compliance audit. Thorough and useless.

What I wanted was *judgment*. The kind of review where someone looks at your validation layer and says "this is doing too much work at runtime that could be enforced at the type level" — not because a checklist told them to say that, but because that's how they think about software.

So I tried something different. I named four people:

- **Samuel Colvin** (Pydantic)
- **Sebastián Ramírez** (FastAPI)
- **Charlie Marsh** (Ruff / uv)
- **Michael Kennedy** (Talk Python)

And I assigned each one a review axis. Samuel gets structural correctness. Sebastián gets API design and developer experience. Charlie gets performance and simplicity. Michael gets teachability and real-world fit.

It worked far better than any instruction set I'd written. But the interesting question isn't *that* it worked — it's *why*.

## Names as compressed instruction sets

When you write "review this code for type safety issues," you're giving a task. When you write "Samuel Colvin reviews this code," you're loading a worldview.

The name "Samuel Colvin" — in the context of Python development — isn't just an identifier. It's a compressed reference to a publicly documented set of technical positions: that runtime validation should be declarative, that type annotations should do real work, that data models should enforce invariants at the boundary, that `dict` is not a data structure. He's written millions of words across docs, talks, GitHub issues, and blog posts that collectively define a coherent perspective on how software should be structured.

An LLM that has processed those words doesn't need you to enumerate the perspective. The name *is* the enumeration.

This is semantic compression. You're replacing a paragraph of instructions with a token that expands — inside the model's latent space — into the same information, plus all the contextual nuance that a paragraph couldn't capture.

## Why it works better than instructions

Consider the alternative. To get Samuel's review axis without using his name, you'd need something like:

> Review the code for structural correctness. Focus on type annotations — are they present, accurate, and enforced? Look at data modeling decisions: are validation boundaries explicit? Are invariants encoded in the type system or checked at runtime? Flag any implicit behavior that could be made explicit through better typing. Consider whether the code distinguishes between validated and unvalidated data. Check for places where a dict is used where a typed model would be more appropriate. Evaluate whether the static and structural layer of the code communicates intent clearly to both humans and tools.

That's 90 words. It's also incomplete. It doesn't capture the *taste* — the instinct for when a type annotation is load-bearing versus ceremonial, or the judgment call about when validation belongs at the boundary versus inline. Those are things you learn from reading someone's work over years, not from reading a paragraph of instructions.

The name compresses all of that — including the parts you can't articulate — into two words.

## The four-person trick

But there's something else going on beyond compression. Naming four people with *distinct* perspectives creates a structure that a single reviewer prompt can't match.

When you ask an LLM to "review this code," it produces a weighted average of all the code review patterns it's seen. Safe, broad, shallow. When you ask four named people to review it, each one filters the codebase through a different lens. The overlap between their observations is where the real issues are. The disagreements are where the interesting tradeoffs live.

Samuel will notice that your error types aren't enforcing invariants. Sebastián will notice that your function signatures are hard to use correctly. Charlie will notice that you're doing three allocations where one would suffice. Michael will notice that a new team member would stare at this for twenty minutes and still not understand the control flow.

These are four *different* observations about the same code. A single "review this code thoroughly" prompt would give you a flattened version — a little of each, with the edges sanded off. The four-person structure keeps the perspectives distinct, which keeps the observations sharp.

## What makes a name work

Not every name compresses well. The technique works when three conditions are met:

**1. Large public corpus.** The person has written, spoken, or coded enough in public that the model has a rich representation of their positions. Samuel Colvin's GitHub history, Pydantic docs, conference talks, and blog posts collectively form a dense signal. A brilliant engineer who's never published anything won't compress — there's nothing to decompress from.

**2. Coherent perspective.** The person has a *consistent* technical worldview, not just a collection of opinions. Charlie Marsh doesn't just happen to care about performance — his entire body of work (Ruff, uv, the blog posts about Python tooling) reflects a coherent philosophy about eliminating unnecessary complexity. That coherence is what makes the name expand into useful judgment rather than random facts.

**3. Distinctness from the ensemble.** The person's perspective needs to be *different* from what you'd get by default. Asking an LLM to "check for performance issues" gets you the median reviewer's performance observations. Naming Charlie Marsh gets you the *specific* kind of performance thinking that builds a Python linter in Rust and considers "just delete it" a valid optimization strategy.

## The pastiche disclaimer

There's an important caveat in the original prompt: *"This is pastiche — you're approximating their public technical positions, not quoting them."*

This matters. The technique works because it leverages the model's representation of someone's *public* positions — their talks, writing, code, and documented opinions. It doesn't work because the model secretly knows what Samuel Colvin thinks about your specific codebase. It's synthesis, not simulation.

The distinction is practical, not just ethical. If you ask the model to simulate a person's private opinions or predict their behavior in situations they've never publicly addressed, you'll get hallucinated specificity that feels authoritative but isn't. If you ask it to approximate the review lens that their public work implies, you get something genuinely useful — a focused perspective that you couldn't have articulated as instructions.

## The generalization

This isn't just a code review trick. Any time you're writing a prompt that needs *judgment* rather than *task execution*, consider whether there's a public figure whose name compresses the judgment criteria you're trying to specify.

Writing a data modeling prompt? "Design this schema the way the person who wrote *Designing Data-Intensive Applications* would" loads a very specific set of priorities about consistency, durability, and schema evolution — priorities you'd struggle to enumerate completely in instructions.

Writing a technical writing prompt? Naming a specific author whose style you admire compresses not just grammar preferences but structural choices, level of assumed knowledge, and the ratio of explanation to assertion.

The technique scales with the model's training data. As public figures produce more work, their names become higher-bandwidth compression tokens. A name that was vague in GPT-3 might be precise in GPT-5 — not because the model got smarter, but because there's more signal to compress.

## The punchline

Four Python developers walk into a bar. They order drinks, open your codebase, and start quietly roasting it to each other.

You don't need to tell them what to look for. They already know. That's the whole point.

What you *do* need to tell them is the ground rules — be specific, cite lines, tag severity, don't inflate nits. The judgment is in the names. The structure is in the prompt.

Two words per reviewer. Eight words total. Expanding into four distinct, coherent, technically grounded perspectives on your code.

That's compression.
