# Regex Validators

Filter extracted spans to match expected patterns, cutting false positives. **Local models only.**
Mirrors
[tutorial/5-validator.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/5-validator.md).

## Quick start

```python
from gliner2 import AutoExtractor, RegexValidator

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")

email_validator = RegexValidator(r"^[\w\.-]+@[\w\.-]+\.\w+$")
schema = (
    extractor.create_schema()
    .structure("contact")
        .field("email", dtype="str", validators=[email_validator])
)
```

Validators also work on `.entities(...)` config dicts, not just `.structure().field(...)`:
`.entities({"label": {"description": "...", "validators": [validator]}})`. Verified: excluding
bare section-number references (`RegexValidator(r"^(Section\s+)?\d+(\.\d+)*\.?$", exclude=True)`)
on an entity label correctly dropped `"Section 8.4"` while a real baseline call without the
validator kept it — see [long-context.md](long-context.md) for the long-document use case.

## Parameters

| Param | Meaning |
|---|---|
| `pattern` | Regex pattern (string or compiled `Pattern`) |
| `mode` | `"full"` (exact match, default) or `"partial"` (substring match) |
| `exclude` | `False` (keep matches, default) or `True` (drop matches) |
| `flags` | Regex flags like `re.IGNORECASE` (string patterns only) |

## Examples

```python
# Email — exact match
RegexValidator(r"^[\w\.-]+@[\w\.-]+\.\w+$")
# "Contact: john@company.com, not-an-email, jane@domain.org" -> ['john@company.com', 'jane@domain.org']

# US phone — partial match (allows surrounding text)
RegexValidator(r"\(\d{3}\)\s\d{3}-\d{4}", mode="partial")
# "Call (555) 123-4567 or 5551234567" -> ['(555) 123-4567']

# URL only
RegexValidator(r"^https?://", mode="partial")
# "Visit https://example.com or www.site.com" -> ['https://example.com']

# Exclude test/demo data
import re
RegexValidator(r"^(test|demo|sample)", exclude=True, flags=re.IGNORECASE)
# "Products: iPhone, Test Phone, Samsung Galaxy" -> ['iPhone', 'Samsung Galaxy']

# Length constraint
RegexValidator(r"^.{5,50}$")
```

## Chaining multiple validators (all must pass)

```python
username_validators = [
    RegexValidator(r"^[a-zA-Z0-9_]+$"),                              # alphanumeric + underscore
    RegexValidator(r"^.{3,20}$"),                                    # 3-20 chars
    RegexValidator(r"^(?!admin)", exclude=True, flags=re.IGNORECASE),  # not "admin"
]
schema = (
    extractor.create_schema()
    .structure("user")
        .field("username", dtype="str", validators=username_validators)
)
# "Users: ab, john_doe, user@domain, admin, valid_user123" -> ['john_doe', 'valid_user123']
```

## Common pattern cheat sheet

| Use case | Pattern | Mode |
|---|---|---|
| Email | `r"^[\w\.-]+@[\w\.-]+\.\w+$"` | full |
| Phone (US) | `r"\(\d{3}\)\s\d{3}-\d{4}"` | partial |
| URL | `r"^https?://"` | partial |
| Numbers only | `r"^\d+$"` | full |
| No spaces | `r"^\S+$"` | full |
| Min length | `r"^.{5,}$"` | full |
| Alphanumeric | `r"^[a-zA-Z0-9]+$"` | full |

## Best practices / performance notes

1. Use the most specific pattern possible — fewer false positives.
2. Test regexes before deployment; validators fail silently (excluded span, no error).
3. Chain multiple simple validators rather than one complex one.
4. Set `re.IGNORECASE` explicitly when case shouldn't matter.
5. Validators run **after** span extraction, **before** output formatting; multiple validators
   short-circuit at the first failure. Compiled patterns are cached automatically.
