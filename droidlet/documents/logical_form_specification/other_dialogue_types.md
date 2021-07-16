Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# More Dialogue Types #
## Noop Dialogue Type ##
```
{ "dialogue_type": "NOOP"}
```

## Get Memory Dialogue Type ##
```
{
  "dialogue_type": "GET_MEMORY",
  "filters": <FILTERS>,
  "replace": true
}
```
## Put Memory Dialogue Type ##
```
{
  "filters": <FILTERS>,
  "upsert" : {
      "memory_data": {
        "memory_type": "REWARD" / "TRIPLE",
        "reward_value": "POSITIVE" / "NEGATIVE",
        "triples": [{"pred_text": "has_x", "obj_text": {"fixed_value" : text} / span}]
      } }
}
```
where `has_x` is one of : `has_tag`, `has_colour`, `has_size`
