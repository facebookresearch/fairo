"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# Notes : this tool assumes that the input will be:
# command \t highlighted command \t action_name \t reference_object \t comparison

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
var ref_obj_child = document.getElementById("ref_child_name_div").textContent;
s = child_name + "_" + ref_obj_child;
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
        <p><b>"destroy the small shiny cube <span style='background-color: #FFFF00'>farthest from here</span>"</b>
        <ul>
        <li>Select 'The value is ranked.' </li>
        <li>Select 'The value is maximum / largest / farthest measure of some kind'</li>
        <li>For 'Select the property or measure of what's being compared.' select 'Distance to / from a given location'</li>
        <li>Now select 'Where the speaker is standing'</li>
        </ul></p>
        <p><b>"fill the <span style='background-color: #FFFF00'>third</span> black hole <span style='background-color: #FFFF00'>on your right </span>"</b>
        <ul>
        <li>Select 'The value is ranked.' </li>
        <li>Select 'The value is minimum / smallest / closest measure of some kind.'</li>
        <li>Select 'Relative direction or positioning from agent' for 'Select the property or measure of what's being compared.'</li>
        <li>Now select 'Right or towards the east direction'</li>
        <li>Select 'Third in the ordered list' for 'What is the position of the property or measure in the ranked list'</li>
        </ul>
        </p>
        <p><b>"destroy the building that is <span style='background-color: #FFFF00'>more than 10 blocks high</span>"</b>
        <li>Select 'The value is being compared to a fixed number.'</li>
        <li>For 'The value is greater than some number' select 'The value is greater than some number'</li>
        <li>For 'Select the property or measure of what's being compared.' select 'height'</li>
        <li>Now select '10' for the 'words for the number being compared against' </li>
        </ul>
        </p>
      </div>
    </div>

    <div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <b>Command: </b><b id="command"></b></div>
    <div id='child_name_div' style='display:none'></div>
    <div id='ref_child_name_div' style='display:none'></div>

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

    from render_questions_for_tools.render_questions_tool_C import *
    from question_flow_in_tools.question_flow_for_tool_D import *

    # all actions with reference objects
    # reference objects with filters and location
    initial_content = (
        "Please give more details about the comparison or ranking of the highlighted property"
    )
    print("""
    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
    <HTMLContent><![CDATA[
    """)
    print(BEFORE)
    print("""
        <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
        <form name='mturk_form' method='post' id='mturk_form' action='https://workersandbox.mturk.com/mturk/externalSubmit'><input type='hidden' value='' name='assignmentId' id='assignmentId'/>
    """)
    # TODO: check if we even need action and child names here at all
    child = "reference_object"
    ref_obj_child = "comparison"
    question = get_questions(child, ref_obj_child)

    if question is not None:
        id_value = child + "_" + ref_obj_child
        render_output = (
            """<div style='font-size:16px;display:block' id='"""
            + id_value
            + """'> <b>"""
            + initial_content
            + """</b>"""
        )
        if type(question) == list:
            for i, q in enumerate(question):
                render_output += render_q(q, "root", show=True)
            render_output += """</div><br><br>"""
            print(render_output)
        else:
            render_output += render_q(question, "root", show=True)
            render_output += """</div><br><br>"""
            print(render_output)

    print("""
     <p><input type='submit' id='submitButton' value='Submit' /></p></form>
        <script language='Javascript'>
        turkSetAssignmentID();
        const queryString = window.location.search;
        console.log(queryString);
        let urlParams = new URLSearchParams(queryString);
        const highlightRange = JSON.parse(urlParams.get("highlight_words"))
        console.log(highlightRange)
        const command = urlParams.get('command');
        console.log(command)
        commandWords = command.split(" ")
        var commandHTML = ""
        for (i = 0; i < commandWords.length; i++) {
          isHighlight = false;
          for (j = 0; j < highlightRange.length; j++) {
            currRange = highlightRange[j]
            highlightStart = currRange[0]
            highlightEnd = currRange[1]
            if (i >= highlightStart && i <= highlightEnd) {
                console.log(commandWords[i])
                commandHTML += ("<span style='background-color: #FFFF00'> " + commandWords[i] + " </span>")
                isHighlight = true;
                break;
            }
          }
          if (!isHighlight) {
            commandHTML += (commandWords[i] + " ")
          }
        }
        
        document.getElementById("command").innerHTML=commandHTML;
        const child = urlParams.get('child');
        document.getElementById("child_name_div").innerHTML=child;
        const ref_child = urlParams.get('ref_child');
        document.getElementById("ref_child_name_div").innerHTML=ref_child;

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
