# Question Patterns

These JSON files describe how the offline assistant should ask questions for
different subjects. They provide question wording and types; answers are still
copied from the source TXT/PDF text so the system does not invent facts.

`catalog.json` contains 1,000 additional educational question pattern types.
For each source sentence, the generator deterministically selects eight
patterns from `catalog.json` first, followed by four subject-specific patterns.
Every generated Q&A record includes `pattern_source` and `pattern_type`, so it
is clear which pattern produced the question. This provides variety without
making results random between runs.

The converter detects a subject from the PDF folder name:

- `english/`
- `science/`
- `mathematics/` or `math/`
- `social_science/` or `social science/`
- any other folder uses `general.json`

Add or edit patterns here, then run:

```bash
./run.sh text
./run.sh convert
```
