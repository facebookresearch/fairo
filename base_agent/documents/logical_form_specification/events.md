Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# EVENT draft #
The "remove" key describes when the even should be deactivated/removed from memory; the "control" clause describes when the event fires.  events are siblings of "action_sequence" and "dialogue_type"
```
EVENT = "event": {"action_sequence" : [ACTION, ...., ACTION],
                  "control": CONTROL,
                  "spatial_control" : SPATIAL_CONTROL
                  "remove": CONDITION}
```
