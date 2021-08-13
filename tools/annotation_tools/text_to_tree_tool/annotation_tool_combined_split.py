"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
BEFORE = """
<!-- Bootstrap v3.0.3 -->
<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />
<section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px;
  font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
<div class="row col-xs-12 col-md-12"><!-- Instructions -->
<div class="panel panel-primary">
<div class="panel-heading"><strong>Instructions</strong></div>

<div class="panel-body" style="font-size:14px;">
<h1><strong>Split a composite command into individuals.</strong></h1>

<p>Please help us split a command into individual single commands. The command shown to you here is given to an AI assistant to help out a player in the game of Minecraft. You will be show a command that possibly implies a sequence or list of single commands and your task is to give us single complete actions that are intended by the command shown to you.</p>
&nbsp;

<p>Few valid examples below:</p>

<p>For <b>&quot;hey bot please build a house and a cube&quot;</b> the answer is the following:</p>

<ul>
	<li>&quot;hey bot please build a house&quot; and</li>
	<li>&quot;hey bot please build a cube&quot;</li>
</ul>

<p>&nbsp;</p>

<p>For <b>&quot;build a castle and then come back here&quot;</b> the answer is the following:</p>

<ul>
	<li>&quot;build a castle&quot; and</li>
	<li>&quot;come back here&quot;</li>
</ul>

<p>&nbsp;</p>

<p>For <b>&quot;destroy the roof and build a stone ceiling in its place&quot;</b> the answer is the following:</p>

<ul>
	<li>&quot;destroy the roof&quot; and</li>
	<li>&quot;build a stone ceiling in its place&quot;</li>
</ul>

<p>&nbsp;</p>

<p>For <b>&quot;move to the door and open it&quot;</b> the answer is the following:</p>

<ul>
	<li>&quot;move to the door&quot; and</li>
	<li>&quot;open the door&quot;</li>
</ul>

<p>&nbsp;</p>

<p>For <b>&quot;i want you to undo the last two spawns and try again with new spawns&quot; </b></p>

<ul>
	<li>&quot;undo the last two spawns&quot; and</li>
	<li>&quot;do a new spawn&quot;</li>
</ul>
<b>Note that: &quot;do a new spawn&quot; is a rewrite of &quot;and try again with new spawns&quot; to make that sub-command clear when seen in isolation.</b>

<p>&nbsp;</p>

<p>Note that:<br />
<b>1. Some commands might have more than two splits. We&#39;ve given you two more optional boxes.</b><br />
<b>2. Make sure that the commands you enter in text boxes are single and complete sentences by their own.</b><br />
<b>3. You might need to rewrite some commands when you split them, to make them clear in isolation.</b></p>
</div>
</div>
<div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
        
<h2><strong>Command:</strong> <b id="sentence"></b></h2> 
</div>
<!-- Content Body -->

<section>

<section>
<fieldset>
<div class="input-group"><span style="font-family: verdana, geneva, sans-serif;font-size: 18px;">The individual commands.&nbsp;</span>

<p>Command 1<textarea class="form-control" cols="150" name="command_1" rows="2"></textarea></p>

<p>Command 2<textarea class="form-control" cols="150" name="command_2" rows="2"></textarea></p>

<p>Command 3 (optional)<textarea class="form-control" cols="150" name="command_3" rows="2"></textarea></p>

<p>Command 4 (optional)<textarea class="form-control" cols="150" name="command_4" rows="2"></textarea></p>
</div>
</fieldset>
"""

AFTER = """
    </section>
    <!-- End Content Body -->

  </div>
</section>

<style type="text/css">fieldset { padding: 10px; background:#fbfbfb; border-radius:5px; margin-bottom:5px; }
</style>
<script src="https://code.jquery.com/jquery.js"></script>
<script src="https://netdna.bootstrapcdn.com/bootstrap/3.0.3/js/bootstrap.min.js"></script>

"""

if __name__ == "__main__":
    # XML: yes, gross, I know. but Turk requires XML for its API, so we deal
    print(
        """
    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
    <HTMLContent><![CDATA[
    """
    )


    print(BEFORE)
    print(
        """
        <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
        <form name='mturk_form' method='post' id='mturk_form' action='https://workersandbox.mturk.com/mturk/externalSubmit'><input type='hidden' value='' name='assignmentId' id='assignmentId'/>
    """
    )
    print(
        """
        <p><input type='submit' id='submitButton' value='Submit' /></p></form>
        <script language='Javascript'>
        turkSetAssignmentID();
        const queryString = window.location.search;
        console.log(queryString);
        let urlParams = new URLSearchParams(queryString);
        const sentence = urlParams.get('sentence');
        document.getElementById("sentence").innerHTML=sentence;
        </script>
        """
    )
    print(AFTER)
    print(
        """
      ]]>
  </HTMLContent>
  <FrameHeight>600</FrameHeight>
  </HTMLQuestion>
    """
    )
