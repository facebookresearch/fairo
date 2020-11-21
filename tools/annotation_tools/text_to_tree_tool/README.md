# Tree to Text tool

This is a Turk tool that is used to annotate English sentences with their
corresponding logical forms.

There is a series of multiple-choice questions answers to which might produce
additional questions, in a structure that mostly mirrors the structure of our nested
dictionaries.

The complete annotation is a two step process:
- `annotation_tool_1` : This tool queries the user for top-level tree structure. Specifically,
it asks about the intended action, words of the sentence that correspond to the respective
children of the action and any repeat patterns.
- `annotation_tool_2`: This tool goes over the children. Specifically, it queries
the users about the highlighted words of subcomponents that came from the first tool
to determine specific properties like name, colour, size etc.

An example:
For the sentence : ```build a red cube here```
The first tool gets a sense of the action which is `Build` in this case, the words for
what needs to be built (`red cube`) and words for where the construction will
happen(`here`).

The second tool now queries the user for:
- The name, colour, size, height etc of the thing that needs to be
built(`red cube` in build a **red cube** here).
- The specifics of the location where the construction will
happen(`here` in build a **red cube** here), by asking questions about whether
the location is relative to something, described using words that are reference to a location etc.

## Render the html
- To see the first tool:
```
 python annotation_tool_1.py > step_1.html
```
- To see the second tool:

```
python annotation_tool_2.py > step_2.html
```
and then open these html file sin browser.

### Note
The second annotation tool needs some input to dynamically render content, most divs
are hidden by default and this is closely stitched to our Turk use case.
But you can do the following to see all questions:
In `annotation_tool_2.py` just change the `display` :
```
render_output = (
            """<div style='font-size:16px;display:none' id='"""
            + id_value
            + """'> <b>"""
            + sentence
            + """</b>"""
        )
```
from `display:none` to `display:block`.


## The Main Files

- `annotation_tool_1.py` and ``annotation_tool_2.py``: The main executables. Run
  these scripts to produce the HTML files that can be copied into the Turk
  editor, or opened in Chrome to view.

- `question_flow_for_step_1.py` and `question_flow_for_step_1.py`: A series of
  JSONs describing the respective questions. This is the file you'll be editing
  if you want to change the questions.

- `render_questions.py` and `render_questions_tool_2.py`: These files render the
  actual HTML elements using the JSON produced from questions flows.
