"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

MAX_WORDS = 40

CSS_SCRIPT = """
<script>
var node = document.createElement('style');
"""
for i in range(MAX_WORDS):
    CSS_SCRIPT += """
        if (! "${{word{i}}}") {{
            node.innerHTML += '.word{i} {{ display: none }} '
        }}
    """.format(
        i=i
    )
CSS_SCRIPT += """
document.body.appendChild(node);
</script>
"""

JS_SCRIPT = """
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})
var child_name = document.getElementById("child_name_div").textContent;
var action_name = document.getElementById("intent_div").textContent;
var x = null;
var y = null;
var s = null;
x = action_name;
y = child_name;
s = action_name + "_" + child_name;
document.getElementById(s).style.display = "block";

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
        <p>Please help us determine the exact meaning of the <span style='background-color: #FFFF00'>highlighted words</span> in the command shown below.
        The command is given to an AI assistant to help a player in the game of Minecraft.</p>

        <p>You will be answering a series of questions about the <span style='background-color: #FFFF00'><b>highlighted text</b></span>. Each question is either multiple-choice,
         or requires you to select which words in the sentence correspond to which property of the thing.</p>
        <p>
        <b>1. Place your mouse arrow over the questions and options for detailed tips.</b></br>
        <b>2. When selecting the words, please select all words (along with properties of the thing).</b> So in "destroy the blue house" select "blue house" and not just "house"</br>
        <b>3. When answering the questions, remember that you are answering them to find more details about the highlighted words .</b>

        </p>

        <p>Few examples below:</p>
        <p><b>"make a <span style='background-color: #FFFF00'>small red bright cube</span> there"</b>
        <ul>
        <li>Select 'Name', 'Abstract/non-numeric size' and 'Colour' properties from the radios.</li>
        <li>For 'Select all words that indicate the name of the thing to be built' select 'cube'</li>
        <li>For 'Select all words that represent the size' select 'small'</li>
        <li>For 'Select all words that represent the colour' select 'red'</li>
        <li>For 'Some other property not mentioned above' select 'bright'</li>
        </ul></p>
        <p><b>"destroy the house <span style='background-color: #FFFF00'>over there</span>"</b>
        <li>For 'Where should the construction happen?' select "The location is represented using an indefinite noun like 'there' or 'over here'" </li>
        </ul>
        </p>
        <p><b>"go to the cube <span style='background-color: #FFFF00'>behind me</span>"</b>
        <li>For 'Where should the construction happen?' select "Somewhere relative to where the speaker is standing" </li>
        <li>For 'Where (which direction) in relation to where the speaker is standing?' select 'Behind'</li>
        </ul>
        </p>
        <p><b>"complete <span style='background-color: #FFFF00'>that</span>"</b>
        <ul>
        <li>Select "There are words or pronouns that refer to the object to be completed"</li>
        </ul>
        </p>
        <p><b>"go <span style='background-color: #FFFF00'>behind the sheep</span>"</b>
        <ul>
        <li>Select "Somewhere relative to another object(s) / area(s)"</li>
        <li>Select "Behind" for "Where (which direction) in relation to the other object(s)?"</li>
        <li>Select "the sheep" for "Click on all words specifying the object / area relative to which location is given"</li>
        </ul>
        </p>
      </div>
    </div>

    <div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <b>Command: </b> <b id="command"></b></div>
    <div id='intent_div' style='display:none'>build</div>
    <div id='child_name_div' style='display:none'>schematic</div>

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

    from render_questions_for_tools.render_questions_tool_2 import *
    from question_flow_in_tools.question_flow_for_tool_B import *

    initial_content = {
        "build_schematic": "Please specify details of the thing that needs to be built.",
        "build_location": "Please specify details of the location at which the construction will happen.",
        "dig_schematic": "Please specify details of the thing that needs to be dug.",
        "dig_location": "Please specify details of the location at which the digging will happen.",
        "copy_reference_object": "Please specify details of the thing that needs to be copied.",
        "copy_location": "Please specify details of the location at which the copy will be made.",
        "freebuild_reference_object": "Please specify details of the thing that needs to be completed.",
        "move_location": "Please specify details of where the assistant should move.",
        "destroy_reference_object": "Please specify details of the thing that needs to be destroyed.",
        "spawn_reference_object": "Please specify details of what the assistant should spawn / generate",
        "spawn_location": "Please specify details of where the assistant should spawn / place",
        "otheraction_reference_object": "Please specify details of the reference object",
        "otheraction_location": "Please specify details of the location",
        "fill_reference_object": "Please specify details of the thing that should be filled",
        "tag_filters": "Please specify details of the thing that being tagged",
        "tag_tag_val": "Please specify details of the tag",
        "dance_location": "Please specify details of where the dance should happen.",
        "look_facing": "Please specify details of where or how the bot should face.",
        "point_facing": "Please specify details of where or how the bot should point.",
        "turn_facing": "Please specify details of where or how the bot should turn its body. ",
        "get_receiver_location": "Please specify details of where the object should be brought to.",
    }

    optional_words = {
        "build": "construction",
        "copy": "copying",
        "spawn": "spawn",
        "dig": "digging",
        "look": "face",
        "point": "point",
        "turn": "turn its body",
    }
    action_children = {
        "build": ["schematic", "location"],
        "copy": ["reference_object", "location"],
        "freebuild": ["reference_object"],
        "move": ["location"],
        "spawn": ["reference_object", "location"],
        "fill": ["reference_object"],
        "destroy": ["reference_object"],
        "dance": ["location"],
        "dig": ["schematic", "location"],
        "tag": ["filters", "tag_val"],
        "otheraction": ["reference_object", "location"],
        "look": ["facing"],
        "point": ["facing"],
        "turn": ["facing"],
        "get": ["receiver_location"],
    }
    print("""
    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
    <HTMLContent><![CDATA[
    """)
    print(BEFORE)
    print("""
        <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
        <form name='mturk_form' method='post' id='mturk_form' action='https://workersandbox.mturk.com/mturk/externalSubmit'><input type='hidden' value='' name='assignmentId' id='assignmentId'/>
    """)
    for action in action_children.keys():
        for child in action_children[action]:
            question = get_questions(child, action, optional_words.get(action, None))

            if question is not None:
                id_value = action + "_" + child
                sentence = initial_content[id_value]
                render_output = (
                    """<div style='font-size:16px;display:none' id='"""
                    + id_value
                    + """'> <b>"""
                    + sentence
                    + """</b>"""
                )
                if type(question) == list:
                    for i, q in enumerate(question):
                        render_output += render_q(q, "root." + action, show=True)
                    render_output += """</div><br><br>"""
                    print(render_output)
                else:

                    render_output += render_q(question, "root." + action, show=True)
                    render_output += """</div><br><br>"""
                    print(render_output)

    print("""
        <p><input type='submit' id='submitButton' value='Submit' /></p></form>
        <script language='Javascript'>
        turkSetAssignmentID();
        const queryString = window.location.search;
        console.log(queryString);
        let urlParams = new URLSearchParams(queryString);
        const highlightRange = urlParams.get('range').split(",").map(Number);
        console.log(highlightRange)
        highlightStart = highlightRange[0]
        highlightEnd = highlightRange[1]

        const command = urlParams.get('command');
        console.log(command)
        commandWords = command.split(" ")
        var commandHTML = ""
        for (i = 0; i < commandWords.length; i++) {
            if (i >= highlightStart && i <= highlightEnd) {
                console.log(commandWords[i])
                commandHTML += ("<span style='background-color: #FFFF00'> " + commandWords[i] + " </span>")
            } else {
                commandHTML += (commandWords[i] + " ")
            }
        }
        document.getElementById("command").innerHTML=commandHTML;
        const intent = urlParams.get('intent');
        console.log(intent)
        document.getElementById("intent_div").innerHTML=intent;
        const child = urlParams.get('child');
        console.log(child)
        document.getElementById("child_name_div").innerHTML=child;
        var styleNode = document.createElement('style');

        for (i = 0; i < 40; i++) {
          var node = 'word' + i
          if (urlParams.has(node)) {
            const word0 = urlParams.get(node);
            var word_list = document.getElementsByClassName(node);
            Array.prototype.forEach.call(word_list, function(el) {
                el.innerHTML = el.innerHTML.replace(node, word0);
            });
          } else {
            styleNode.innerHTML += '.' + node + ' { display: none } ';
          }
        }
        document.body.appendChild(styleNode);
        </script>
    """)
    print(AFTER)
    print("""
      ]]>
  </HTMLContent>
  <FrameHeight>600</FrameHeight>
  </HTMLQuestion>
    """)
