"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# Notes : this tool assumes that the input will be:
# command \t highlighted command \t action_name \t reference_object

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

# previous JS_SCRIPT with ref_object_child was:
"""
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})
var child_name = document.getElementById("child_name_div").textContent;
var action_name = document.getElementById("intent_div").textContent;
var ref_obj_child = document.getElementById("ref_child_name_div").textContent;
s = action_name + "_" + child_name + "_" + ref_obj_child;
document.getElementById(s).style.display = "block";
"""

JS_SCRIPT = """
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})
var child_name = document.getElementById("child_name_div").textContent;
var action_name = document.getElementById("intent_div").textContent;
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
        <p>Please help us determine more detailed properties of the <span style='background-color: #FFFF00'>highlighted words</span> in the command shown below.
        The command is given to an AI assistant to help a player in the game of Minecraft.</p>

        <p>You will be answering a series of questions about the <span style='background-color: #FFFF00'><b>highlighted text</b></span>. Each question is either multiple-choice,
         or requires you to select which words in the sentence correspond to which property of the thing.</p>
        <p>
        <b>1. Place your mouse arrow over the questions and options for detailed tips.</b></br>
        <b>2. When selecting the words, please select all words (along with properties of the thing).</b> So in "destroy the tidy bright house" select "tidy bright" and not just "tidy"</br>
        <b>3. When answering the questions, remember that you are answering them to find more details about the highlighted words .</b>

        </p>

        <p>Few examples below:</p>
        <p><b>"destroy the <span style='background-color: #FFFF00'>small shiny cube farthest from here</span>"</b>
        <ul>
        <li>Select 'Name', 'Abstract/non-numeric size',  'Some property of this object is being compared or ranked' and 'Some other property not covered by anything above' properties from the radios.</li>
        <li>For 'What is the name of the object that should be destroyed?' select 'cube'</li>
        <li>For 'What is the size?' select 'small'</li>
        <li>For 'Some property of this object is being compared or ranked' select 'farthest from here'</li>
        <li>For 'Some other property not covered by anything above' select 'shiny'</li>
        </ul></p>
        <p><b>"fill the <span style='background-color: #FFFF00'>third black hole on your right </span>"</b>
        <ul>
        <li>Select 'Name', 'Colour' and  'Some property of this object is being compared or ranked' properties from the radios.</li>
        <li>For 'What is the name of the object that should be destroyed?' select 'hole'</li>
        <li>For 'What is the colour?' select 'black'</li>
        <li>For 'Some property of this object is being compared or ranked' select 'third on your right'</li>
        </ul>
        </p>
        <p><b>"go inside <span style='background-color: #FFFF00'>the tallest building</span>"</b>
        <li>Select 'Name' and  'Some property of this object is being compared or ranked' properties from the radios.</li>
        <li>For 'What is the name of the reference object' select 'building'</li>
        <li>For 'Some property of this object is being compared or ranked' select 'tallest'</li>
        </ul>
        </p>
        <p><b>"complete <span style='background-color: #FFFF00'>that</span>"</b>
        <ul>
        <li>Select "There are words or pronouns that refer to the object to be completed"</li>
        </ul>
        </p>
      </div>
    </div>

    <div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <b>Command: </b><b id='command'></b></div>
    <div id='intent_div' style='display:none'></div>
    <div id='child_name_div' style='display:none'></div>
    
    <!-- Content Body -->
    <section>
"""

"""
# NOTE: if ref_object_child is in, add the following above:
<div id='ref_child_name_div' style='display:none'>${ref_child}</div>
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

    from render_questions_for_tools.render_questions_tool_C import *
    from question_flow_in_tools.question_flow_for_tool_C import *

    # all actions with reference objects
    # reference objects with filters and location
    initial_content = {
        # COPY
        "copy_reference_object": "Please specify details of the thing that needs to be copied.",
        "copy_reference_object_filters": "Please give more details about the comparison being made.",
        "copy_reference_object_location": "Please specify details of location of what needs to be copied.",
        # FREEBUILD
        "freebuild_reference_object": "Please specify details of the thing that needs to be completed.",
        "freebuild_reference_object_filters": "Please give more details about the comparison being made.",
        "freebuild_reference_object_location": "Please specify details of location of what needs to be completed.",
        # DESTROY
        "destroy_reference_object": "Please specify details of the thing that needs to be destroyed.",
        "destroy_reference_object_filters": "Please give more details about the comparison being made.",
        "destroy_reference_object_location": "Please specify details of location of what needs to be destroyed.",
        # SPAWN
        "spawn_reference_object": "Please specify details of what the assistant should spawn / generate",
        "spawn_reference_object_filters": "Please give more details about the comparison being made.",
        "spawn_reference_object_location": "Please specify details of location of what needs to be spawned / generated.",
        # OTHER ACTION
        "otheraction_reference_object": "Please give more details about the reference object",
        "otheraction_reference_object_filters": "Please give more details about the comparison being made.",
        "otheraction_reference_object_location": "Please specify details of location of the reference object",
        # FILL
        "fill_reference_object": "Please specify details of the thing that should be filled",
        "fill_reference_object_filters": "Please give more details about the comparison being made.",
        "fill_reference_object_location": "Please specify details of location of what should be filled",
        # TAG
        "tag_filters": "Please specify details of the thing that is being tagged",
        # GET
        "get_reference_object": "Please specify details of the thing that will be brought",
        "get_receiver_reference_object": "Please specify details of the thing where the object will be brought to",
    }

    optional_words = {"copy": "copying", "spawn": "spawn"}

    action_children = {
        "copy": ["reference_object"],
        "freebuild": ["reference_object"],
        "spawn": ["reference_object"],
        "fill": ["reference_object"],
        "destroy": ["reference_object"],
        "otheraction": ["reference_object"],
        "get": ["reference_object", "receiver_reference_object"],
    }

    # NOTE: For now, we allow location in built in the tool.
    # For filters: Turkers can select the words corresponding to the comparison -> send to
    # comparsion_tool_filters.py
    # For location: Turkers can select location type and spans, if location_type
    #               is reference_object, we get the span and reuse this tool further
    #               to get annotations.
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
        for child in action_children[action]:  # only ref object here
            ref_obj_child = ""
            # NOTE: if ref_obj_child is allowed, add : for ref_obj_child in ["", "filters", "location"]:
            question = get_questions(
                child, action, ref_obj_child, optional_words.get(action, None)
            )

            if question is not None:
                id_value = action + "_" + child  # + "_" + ref_obj_child
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
                        render_output += render_q(q, "root." + action + "!" + child, show=True)
                    render_output += """</div><br><br>"""
                    print(render_output)
                else:
                    render_output += render_q(question, "root." + action + "!" + child, show=True)
                    render_output += """</div><br><br>"""
                    print(render_output)

    print("""
     <p><input type='submit' id='submitButton' value='Submit' /></p></form>
        <script language='Javascript'>
        turkSetAssignmentID();
        const queryString = window.location.search;
        console.log(queryString);
        let urlParams = new URLSearchParams(queryString);
        const highlightRange = JSON.parse(urlParams.get("highlight_words"))[0]
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
        document.getElementById("intent_div").innerHTML=intent;
        const child = urlParams.get('child');
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
