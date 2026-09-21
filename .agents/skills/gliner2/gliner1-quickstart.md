# GLiNER (v1): Quickstart

See [gliner1-intro.md](gliner1-intro.md) first for what GLiNER is and how it relates to GLiNER2.
This is the minimal path from `pip install` to a first prediction, mirroring the
[GLiNER quickstart](https://urchade.github.io/GLiNER/quickstart.html).

```bash
pip install gliner
```

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

text = """
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC
interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of
chairman, chief executive officer, president and chief software architect, while also being the
largest individual shareholder until May 2014.
"""

labels = ["founder", "computer", "software", "position", "date"]
entities = model.predict_entities(text, labels)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

```text
Bill Gates => founder
Paul Allen => founder
April 4, 1975 => date
BASIC interpreters => software
Altair 8800 => computer
chairman => position
chief executive officer => position
president => position
chief software architect => position
largest individual shareholder => position
May 2014 => date
```

That's the whole API surface for the common case: `from_pretrained(checkpoint)` +
`predict_entities(text, labels)`. Everything else in this package — batching, label descriptions,
thresholds, multi-label, architecture selection, performance tuning, streaming, training, export,
serving — is additive on top of this call. See [gliner1-usage.md](gliner1-usage.md) next.
