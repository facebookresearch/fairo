# Example 12: Importing other msetup

In this example, we show two ways of importing process definitions from other `msetup.py`.

The first, is to directly import the module.

The top level `msetup.py` is able to import bob's processes with
```py
import bob.msetup
```

The second method, where refering to modules directly might be painful, we can use the helper method
```py
mrp.import_msetup("../alice")
```
as is done in bob's `msetup.py`
