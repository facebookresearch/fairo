"""
Copyright (c) Facebook, Inc. and its affiliates.
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
        <h1><strong>Split a composite command into individuals.</strong></h1>
        <p>Please help us split a command into individual single commands.
        The command shown to you here is given to an AI assistant to help out a player in the game of Minecraft.
        You will be show a command that possibly implies a sequence or list of single commands and your task is to give us
        single complete actions that are intended by the command shown to you.</p>
        </br>
        </br>
        <p>Few valid examples below: </p>
        <p>For <b>"hey bot please build a house and a cube"</b>
        the answer is the following:
        <ul>
           <li>"hey bot please build a house" and </li>
           <li>"hey bot please build a cube"</li>
        </ul>
        </p>

        <p>For <b>"build a castle and then come back here"</b>
        the answer is the following:
        <ul>
           <li>"build a castle" and </li>
           <li>"come back here"</li>
        </ul>
        </p>
        <p>For <b>"destroy the roof and build a stone ceiling in its place"</b>
        the answer is the following:
        <ul>
           <li>"destroy the roof" and </li>
           <li>"build a stone ceiling in its place"</li>
        </ul>
        </p>
        <p>For <b>"move to the door and open it"</b>
        the answer is the following:
        <ul>
           <li>"move to the door" and </li>
           <li>"open the door"</li>
        </ul>
        </p>
        <p>For <b>"i want you to undo the last two spawns and try again with new spawns" </b>
        <ul>
            <li>"undo the last two spawns" and </li>
            <li>"do a new spawn"</li>
        </ul>
        <b>Note that: "do a new spawn" is a rewrite of "and try again with new spawns" to make that sub-command clear when seen in isolation.</b>
        </p>

        <p> Note that:</br>
        <b>1. Some commands might have more than two splits. We've given you two more optional boxes.</b></br>
        <b>2. Make sure that the commands you enter in text boxes are single and complete sentences by their own.</b></br>
        <b>3. You might need to rewrite some commands when you split them, to make them clear in isolation.</b>
        </p>
      </div>
    </div>

    <div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <h2><strong>Command:</strong> ${sentence}</h2>
    </div>

    <!-- Content Body -->
    <section>
"""

# AFTER = """
#     </section>
#     <!-- End Content Body -->
#
#   </div>
# </section>
#
# <style type="text/css">
#     fieldset {{
#     padding: 10px;
#     font-family: Georgia;
#     font-size: 14px;
#     background: #fbfbfb;
#     border-radius: 5px;
#     margin-bottom: 5px;
#     }}
#     .tooltip {{
#     font-family: Georgia;
#     font-size: 18px;
#     }}
#     .tooltip .tooltip-inner {{
#       background-color: #ffc;
#       color: #c00;
#       min-width: 250px;
#     }}
# </style>
#
# {CSS_SCRIPT}
#
# <script src="https://code.jquery.com/jquery.js"></script>
# <script src="https://netdna.bootstrapcdn.com/bootstrap/3.0.3/js/bootstrap.min.js"></script>
#
# <script>{JS_SCRIPT}</script>
# """.format(
#     CSS_SCRIPT=CSS_SCRIPT, JS_SCRIPT=JS_SCRIPT
# )

BETWEEN = """
    <section>
    <fieldset>
    <div class="input-group"><span style="font-family: verdana, geneva, sans-serif;font-size: 18px;">The individual commands.&nbsp;</span>

    <p>Command 1 <textarea class="form-control" cols="150" name="command_1" rows="2"></textarea></p>
    <p>Command 2 <textarea class="form-control" cols="150" name="command_2" rows="2"></textarea></p>
    <p>Command 3 (optional)<textarea class="form-control" cols="150" name="command_3" rows="2"></textarea></p>
    <p>Command 4 (optional)<textarea class="form-control" cols="150" name="command_4" rows="2"></textarea></p>

    </div>
    </fieldset>
    </section>
    <!-- End Content Body --></div>
    </div>
    </section>
    <style type="text/css">fieldset { padding: 10px; background:#fbfbfb; border-radius:5px; margin-bottom:5px; }
    </style>
"""
if __name__ == "__main__":

    print(
        BEFORE,
        BETWEEN
        # render_questions.render_q(Q_ACTION, "root", show=True),
        # render_questions.render_q(Q_ACTION_LOOP, "root", show=True),
        # AFTER,
    )
