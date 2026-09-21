# GLiNER (v1): Creating Custom Architectures

How to extend GLiNER with a new architecture variant by implementing the four component classes
plus a high-level wrapper. Condensed from
[Creating Custom GLiNER Architectures](https://urchade.github.io/GLiNER/add_custom_architecture.html).
Only needed if none of the built-in architectures fit — check
[gliner1-configs-architectures.md](gliner1-configs-architectures.md)'s decision flowchart first.

## Component responsibilities

```
                   High-Level Wrapper (e.g. UniEncoderSpanGLiNER)
              user-facing API (predict_entities, train, ...); model instantiation/management
                                    │
        ┌─────────────┬────────────┼──────────────┬─────────────┐
   Configuration      Model      Processor       Decoder
```

| Component | Purpose | Base class |
|---|---|---|
| Configuration | Hyperparameters and settings | `BaseGLiNERConfig` |
| Model | Neural network (forward pass, loss) | `BaseModel` |
| Processor | Preprocessing and tokenization | `BaseProcessor` |
| Decoder | Model outputs → entity predictions | `BaseDecoder` |
| High-Level Wrapper | User-facing API and orchestration | `BaseGLiNER` |

## 1. Configuration

```python
from gliner.config import BaseGLiNERConfig

class MyCustomConfig(BaseGLiNERConfig):
    def __init__(self, pooling_strategy: str = "mean", use_cnn: bool = False,
                 cnn_filters: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.pooling_strategy = pooling_strategy
        self.use_cnn = use_cnn
        self.cnn_filters = cnn_filters
        self.model_type = "custom_gliner"   # identifies the architecture for dispatch
```

## 2. Model

```python
from gliner.modeling.base import BaseModel
from gliner.modeling.outputs import GLiNERBaseOutput
import torch

class MyCustomModel(BaseModel):
    def __init__(self, config, from_pretrained=False, cache_dir=None):
        super().__init__(config, from_pretrained, cache_dir)
        self.encoder = ...      # your encoder
        self.classifier = ...   # your classifier

    def forward(self, input_ids, attention_mask, **kwargs):
        embeddings = self.encoder(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        loss = self.loss(logits, kwargs["labels"], **kwargs) if "labels" in kwargs else None
        return GLiNERBaseOutput(logits=logits, loss=loss)

    def loss(self, logits, labels, **kwargs):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def get_representations(self, input_ids, attention_mask, **kwargs):
        return self.encoder(input_ids, attention_mask)   # for analysis/debugging
```

## 3. Processor

```python
from gliner.data_processing import BaseProcessor
import torch

class MyCustomProcessor(BaseProcessor):
    def preprocess_example(self, tokens, ner, classes_to_id):
        return {"tokens": tokens, "labels": self._create_labels(tokens, ner, classes_to_id)}

    def _create_labels(self, tokens, ner, classes_to_id):
        ...  # build the label tensor from entity annotations

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        return {
            "tokens": [ex["tokens"] for ex in batch],
            "labels": torch.stack([ex["labels"] for ex in batch]),
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        tokenized = self.tokenize_inputs(batch["tokens"], batch["classes_to_id"])
        if prepare_labels:
            tokenized["labels"] = self.create_labels(batch)
        return tokenized
```

## 4. Decoder

```python
from gliner.decoding import BaseDecoder
import torch

class MyCustomDecoder(BaseDecoder):
    def decode(self, tokens, id_to_classes, model_output, threshold=0.5, **kwargs):
        probs = torch.sigmoid(model_output)
        predictions = []
        for sample_probs in probs:
            sample_preds = []
            # extract entities from probabilities above threshold →
            # standard tuple format: (start, end, label, score)
            predictions.append(sample_preds)
        return predictions
```

## 5. High-level wrapper (ties it together)

```python
from gliner import BaseGLiNER
from transformers import AutoTokenizer

class MyCustomGLiNER(BaseGLiNER):
    config_class = MyCustomConfig
    model_class = MyCustomModel
    data_processor_class = MyCustomProcessor
    decoder_class = MyCustomDecoder
    # data_collator_class = MyCustomDataCollator   # if needed

    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        return self.model_class(config, backbone_from_pretrained, cache_dir, **kwargs)

    def _create_data_processor(self, config, cache_dir, tokenizer=None, **kwargs):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
        return self.data_processor_class(config, tokenizer, None)

    def resize_embeddings(self):
        pass   # implement if you add special tokens

    def inference(self, texts, labels, **kwargs):
        return super().inference(texts, labels, **kwargs)   # or customize

    def evaluate(self, test_data, **kwargs):
        return super().evaluate(test_data, **kwargs)         # or customize
```

Once registered this way, `MyCustomGLiNER` gets the same `from_pretrained`/`from_config`/
`predict_entities`/`train_model` surface as the built-in architectures — callers don't need to know
it's custom. Cross-check your `model_type` string against the built-ins in
[gliner1-configs-architectures.md](gliner1-configs-architectures.md) to avoid a collision.
