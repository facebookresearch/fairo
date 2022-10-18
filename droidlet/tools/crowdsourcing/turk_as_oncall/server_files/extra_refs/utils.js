/*
   Copyright (c) Facebook, Inc. and its affiliates.
 * utils.js contains a variety of click handlers and timing analytics for craftassist_task
 */

// Remove the stylesheet inherited from Mephisto by default
$('link[rel=stylesheet][href~="https://cdn.jsdelivr.net/npm/bulma@0.8.2/css/bulma.min.css"]').remove();

var timerStarted = false;
var timerStopped = false;
var ratingComplete = false;
var commandIssued = false;
var clickedElements = new Array();

// recordClick logs user actions as well as messages from the dashboard to the Mephisto form
function recordClick(ele) {
  if (ele === "timerON") {  // Don't allow these to get pushed more than once 
    if (!timerStarted) {clickedElements.push({id: ele, timestamp: Date.now()});}
    timerStarted = true;
  }
  if (ele === "timerOFF") {  
    if (!timerStopped) {clickedElements.push({id: ele, timestamp: Date.now()});}
    timerStopped = true;
    checkSubmitDisplay();  // Check to allow submission if all qualifications are met
  }
  else { clickedElements.push({id: ele, timestamp: Date.now()}); }

  document.getElementById("clickedElements").value = JSON.stringify(clickedElements);
  // console.log("Clicked elements array: " + JSON.stringify(clickedElements));
}
recordClick("start");

document.getElementsByClassName("btn-default")[0].classList.add("hidden");  // Hide the submit button to start
function checkSubmitDisplay() {
  // Let them submit if they sent at least one command
  clickedElements.forEach(function(click){
    if (click.id === "askActionQuestion") {commandIssued = true}
  })

  //Only display the submit button if the worker has interacted with the dashboard and completed the survey
  if (ratingComplete && commandIssued && timerStopped) {
    document.getElementsByClassName("btn-default")[0].classList.remove("hidden");
    window.scrollTo(0,document.body.scrollHeight);
  }
}


function usabilityRated() {
  ratingComplete = true;
  recordClick("usability-rated");
  checkSubmitDisplay();
}


// Log data from postMessage, and record it if it seems to come from the dashboard
var data;
if (window.addEventListener) {
  window.addEventListener("message", (event) => {
    if (typeof(event.data) === "string") {
      data = JSON.parse(event.data);
    } else data = event.data
    console.log(data);
    if (data.msg) recordClick(data.msg);
  }, false);
}
else if (window.attachEvent) {  // Cross compatibility for old versions of IE
  window.attachEvent("onmessage", (event) => {
    if (typeof(event.data) === "string") {
      data = JSON.parse(event.data);
    } else data = event.data
    console.log(data);
    if (data.msg) recordClick(data.msg);
  });
}

function showHideInstructions() {
  if (document.getElementById("instructionsWrapper").classList.contains("in")){
    for (let ele of document.getElementsByClassName("collapse")){
      if (ele.classList.contains("in")) ele.classList.remove("in");
    }
  }
  else {
    for (let ele of document.getElementsByClassName("collapse")){
      if (ele.classList.contains("in")) continue;
      else ele.classList.add("in");
    }
  }
  recordClick("toggle-instructions");
}

var page = 1;
const numPages = 3;

function incrementPage(val) {
  page += val;
  if (page < 1) page = 1;
  if (page > numPages) page = numPages;
  applyPagination(page);
}

function applyPagination(pg) {
  const pageID = "page-" + pg;
  recordClick(pageID);
  page = pg;

  if (pg === 1) document.getElementById("previous").classList.add("disabled");
  else document.getElementById("previous").classList.remove("disabled");
  if (pg === numPages) {
    document.getElementById("next").classList.add("disabled");
    document.getElementById("close-instructions-btn").style.display = "block";
  }
  else document.getElementById("next").classList.remove("disabled");
  let pgID, btnID;
  for (let i=1; i<=numPages; i++) {
    pgID = "instructions-page-" + i;
    btnID = "pg" + i;
    if (i === pg) {
      document.getElementById(btnID).classList.add("active");
      document.getElementById(pgID).classList.add("in");
    }
    else {
      document.getElementById(btnID).classList.remove("active");
      document.getElementById(pgID).classList.remove("in");
    }
  }
}

//Retrieve and process command list and answer list
function processCommandList() {
  let commands = document.getElementById("commandList").value.split("|");
  let answers = document.getElementById("answerList").value.split("|");

  for (i=1; i<(commands.length + 1); i++) {
    let li = document.createElement("li");
    li.appendChild(document.createTextNode("Command " + i + ": '" + commands[i-1] + "'"));
    li.appendChild(document.createElement("br"));
    li.appendChild(document.createTextNode("Expected Result: " + answers[i-1]));
    document.getElementById("commandPanel").appendChild(li);

    let question = document.createElement("h4");
    question.appendChild(document.createTextNode("Did the assistant respond correctly to command #" + i + ": '" + commands[i-1] + "' --> " + answers[i-1] + ""));
    document.getElementById("survey").appendChild(question);

    let answer = document.createElement("select");
    answer.classList.add("form-control");
    answer.id = "command_" + i;
    answer.name = answer.id;
    let yes = document.createElement("option");
    let no = document.createElement("option");
    yes.value = "yes";
    no.value = "no";
    yes.appendChild(document.createTextNode("Yes"));
    no.appendChild(document.createTextNode("No"));
    answer.appendChild(yes);
    answer.appendChild(no);
    answer.setAttribute('onchange', ("showFeedback(" + i + ")"));
    document.getElementById("survey").appendChild(answer);

    let feedback = document.createElement("input");
    feedback.type = "hidden";
    feedback.classList.add("form-control");
    feedback.id = "command_" + i + "_feedback";
    feedback.name = feedback.id;
    feedback.placeholder = "Please describe what happened instead (required!)";
    document.getElementById("survey").appendChild(feedback);
    document.getElementById("survey").appendChild(document.createElement("br"));
    document.getElementById("survey").appendChild(document.createElement("br"));
    document.getElementById("survey").appendChild(document.createElement("br"));
  }
}
processCommandList();

//Reveal the associated feedback form when a question is answered 'no'
function showFeedback(num) {
  console.log("Show feedback" + num);
  let val = document.getElementById("command_" + num).value;
  let feedback = document.getElementById("command_" + num + "_feedback");
  if (val == "no") {
    feedback.type = "text";
  } else {
    feedback.type = "hidden";
  }
}
