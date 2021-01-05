"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

MAX_WORDS = 30

CSS_SCRIPT = """
<script>
var node = document.createElement('style');
"""
for j in range(1, 4):
    for i in range(MAX_WORDS):
        CSS_SCRIPT += """
            if (! "${{word{j}{i}}}") {{
                node.innerHTML += '.word{j}{i} {{ display: none }} '
            }}
        """.format(
            i=i, j=j
        )
CSS_SCRIPT += """
document.body.appendChild(node);
</script>
"""

JS_SCRIPT = """
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})
"""
BEFORE = """
<!-- Bootstrap v3.0.3 -->

<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />

<section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px;
  font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
  <div class="row col-xs-12 col-md-12">

    <!-- Instructions -->
    <div class="panel panel-primary">
      <div class="panel-heading"><strong>Instructions</strong></div>

      <div class="panel-body" style="font-size:14px;">
        <p><b>Your HIT will will be rejected if you don't answer all three commands before submitting.</b></p>
        <p>Please help us determine the exact meaning of the command shown to you.
        The command is given to an AI assistant to help out a player in the game of Minecraft.</p>

        <p>For each command, you will answer a series of questions. Each question is either multiple-choice,
         or requires you to select which words in the sentence correspond to which components of the command.</p>
        <p>
        <b>1. Place your mouse arrow over the questions and options for detailed tips.</b></br>
        <b>2. When selecting the words, please select all words (along with properties of the thing).</b> So in "destroy the blue house" select "blue house" and not just "house"</br>
        <b>3. Please also note that: </b>some questions are optional, click on "Click if specified" if you think answers to those are mentioned in the command.
        </p>

        <p>Few examples below:</p>
        <p><b>"come"</b>
        <ul>
        <li>For "What action is being requested?", the answer is "Move or walk somewhere"</li>
        </ul></p>
        <p><b>"make two small cubes here"</b>
        <ul>
        <li>"What action is being requested?" -> "Build, make a copy or complete something"</li>
        <li>"Is this an exact copy or duplicate of an existing object?" -> "No". The assistant is asked to "Build a fresh complete, specific object"</li>
        <li>For "Select words specifying what needs to be built" select the words: 'small cubes'</li>
        <li>For "Select words specifying where the construction needs to happen", click on the word: 'here'</li>
        <li>For "How many times should this action be performed?", select "Repeatedly, a specific number of times"
        and then "two" for 'How many times'</li>
        </ul>
        </p>
        <p><b>"dig until you reach water"</b>
        <ul>
        <li>"What action is being requested?" -> "Dig"</li>
        <li>For "How many times should this action be performed?" -> 'Repeated until a certain condition is met'</li>
        <li>For "Until the assistant reaches some object(s) /area" select: "water"</li>
        </ul>
        <b>Note that: repeats may be disguised, for example: 'follow the pig' should be interpreted as "repeat forever: move to the location of the pig".</b>
        </p>
        <p><b>"go to the large pole near the bridge"</b></br>
        <ul>
        <li>"What action is being requested?" -> "Move or walk somewhere"</li>
        <li>"Select words specifying the location to which the agent should move" -> "the large pole near the bridge". </li>
        </ul>
        </p>
      </div>
    </div>
    <!-- Content Body -->
    <section>
"""

AFTER = """
    </section>
    <!-- End Content Body -->

  </div>
</section>

<style type="text/css">
    fieldset {{
    padding: 10px;
    font-family: Georgia;
    font-size: 14px;
    background: #fbfbfb;
    border-radius: 5px;
    margin-bottom: 5px;
    }}
    .tooltip {{
    font-family: Georgia;
    font-size: 18px;
    }}
    .tooltip .tooltip-inner {{
      background-color: #ffc;
      color: #c00;
      min-width: 250px;
    }}
</style>

{CSS_SCRIPT}

<script src="https://code.jquery.com/jquery.js"></script>
<script src="https://netdna.bootstrapcdn.com/bootstrap/3.0.3/js/bootstrap.min.js"></script>

<script>{JS_SCRIPT}</script>
""".format(
    CSS_SCRIPT=CSS_SCRIPT, JS_SCRIPT=JS_SCRIPT
)

if __name__ == "__main__":
    import render_questions

    from question_flow_for_step_1 import *

    print(BEFORE)

    render_output = """<div style='font-size:16px;display:block' id='div_1'>"""
    render_output += """<div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999" id='cmd_1'>
    <b>Command: ${command_1}</b></div>"""

    render_output += render_questions.render_q(Q_ACTION, "root.1", show=True, sentence_id=1)
    render_output += render_questions.render_q(Q_ACTION_LOOP, "root.1", show=True, sentence_id=1)
    render_output += """</div><br><br>"""

    render_output += """<br style=“line-height:50;”>"""
    render_output += """<hr size="10">"""
    render_output += """<div style='font-size:16px;display:block' id='div_2'>"""
    render_output += """<div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <b>Command: ${command_2}</b></div> """
    render_output += render_questions.render_q(Q_ACTION, "root.2", show=True, sentence_id=2)
    render_output += render_questions.render_q(Q_ACTION_LOOP, "root.2", show=True, sentence_id=2)
    render_output += """</div><br><br>"""

    render_output += """<br style=“line-height:50;”>"""
    render_output += """<hr size="10">"""
    render_output += """<div style='font-size:16px;display:block' id='div_3'>"""
    render_output += """<div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <b>Command: ${command_3}</b></div> """
    render_output += render_questions.render_q(Q_ACTION, "root.3", show=True, sentence_id=3)
    render_output += render_questions.render_q(Q_ACTION_LOOP, "root.3", show=True, sentence_id=3)
    render_output += """</div><br><br>"""
    print(render_output)

    print(AFTER)
