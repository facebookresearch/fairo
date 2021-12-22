/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * utils.js contains a variety of click handlers and timing analytics for vision_annotation_task
 */

// Remove the stylesheet inherited from Mephisto by default
$('link[rel=stylesheet][href~="https://cdn.jsdelivr.net/npm/bulma@0.8.2/css/bulma.min.css"]').remove();

var q1Correct = false;
var q2Correct = false;
var q3Correct = false;
var q4Correct = false;
var clickedElements = new Array();

let q1Answer = [[2,0,2]];
let q2Answer = [[-1,0,2],[-1,1,2],[-1,2,2],[0,0,2],[0,1,2],[0,2,2],[1,0,2],[1,1,2],[1,2,2]];
let q4Answer = [[5,0,0],[5,1,0],[4,0,0],[4,1,0],[5,0,-1],[5,1,-1],[4,0,-1],[4,1,-1],[-5,0,0],[-5,1,0],[-4,0,0],[-4,1,0],[-5,0,-1],[-5,1,-1],[-4,0,-1],[-4,1,-1]];

// recordClick logs user actions as well as messages from the dashboard to the Mephisto form
function recordClick(ele) {
  if (ele["q1_output"]) {
    if (JSON.stringify(JSON.parse(ele.q1_output).sort()) === JSON.stringify(q1Answer.sort())){
      q1Correct = true;
    }
    document.getElementById("q1Answer").value = JSON.stringify(q1Correct);
  }
  else if (ele["q2_output"]) {
    if (JSON.stringify(JSON.parse(ele.q2_output).sort()) === JSON.stringify(q2Answer.sort())){
      q2Correct = true;
    }
    document.getElementById("q2Answer").value = JSON.stringify(q2Correct);
  }
  else if (ele["q3_output"]) {
    if (JSON.parse(ele.q3_output) === null){
      q3Correct = true;
    }
    document.getElementById("q3Answer").value = JSON.stringify(q3Correct);
  }
  else if (ele["q4_output"]) {
    if (JSON.stringify(JSON.parse(ele.q4_output).sort()) === JSON.stringify(q4Answer.sort())){
      q4Correct = true;
    }
    document.getElementById("q4Answer").value = JSON.stringify(q4Correct);
  }
  
  clickedElements.push({id: ele, timestamp: Date.now()});
  document.getElementById("clickedElements").value = JSON.stringify(clickedElements);
  console.log("Clicked elements array: " + JSON.stringify(clickedElements));
}
recordClick("start");

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

