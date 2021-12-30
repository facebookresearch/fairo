/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * utils.js contains a variety of click handlers and timing analytics for vision_annotation_task
 */

// Remove the stylesheet inherited from Mephisto by default
$('link[rel=stylesheet][href~="https://cdn.jsdelivr.net/npm/bulma@0.8.2/css/bulma.min.css"]').remove();

var blockMarked = false;
var annotationFinished = false;
var ratingComplete = false;
var feedbackGiven = false;
var clickedElements = new Array();

// recordClick logs user actions as well as messages from the dashboard to the Mephisto form
function recordClick(ele) {
  console.log("Element being processed: " + ele);
  if (typeof(ele) === 'object'){
    let taskID = Object.keys(ele)[0];
    if (taskID.includes("output")) {  // Store the annotation result in Mephisto
      console.log("Recording output for taskID: " + taskID);
      document.getElementById("taskID").value = taskID;
      document.getElementById("markedBlocks").value = ele[taskID];
    }
  }

  if (ele === "block_marked") {  // Check to allow submission if all qualifications are met
    blockMarked = true;
    checkSubmitDisplay();
  }
  if (ele === "annotation_finished") {  // Check to allow submission if all qualifications are met
    annotationFinished = true;
    checkSubmitDisplay();
  }
  clickedElements.push({id: ele, timestamp: Date.now()});
  document.getElementById("clickedElements").value = JSON.stringify(clickedElements);
}
recordClick("start");

// Log data from postMessage, and record it if it seems to come from the iframe
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

// Have the instructions follow the user when scrolling
document.addEventListener('scroll', function(e) {
  let panel = document.getElementById("heading");
  if (window.scrollY > 10) {
    panel.style.position = "fixed"; // sticky won't work b/c 'app' has overflow: hidden
    panel.style.top = "0px";
    panel.style.marginLeft = "5%";
    if (!instructionsCollapsed) {
      document.getElementById("demos").style.marginTop = document.getElementById("heading").offsetHeight + "px";
    }
  } else {
    panel.style.position = "relative";
    document.getElementById("demos").style.marginTop = "0px";
  }
});

// toggle instructions collapse on button click
var instructionsCollapsed = false;
function showHideInstructions() {
  if (document.getElementById("instructionsWrapper").classList.contains("in")){
    instructionsCollapsed = true;
    for (let ele of document.getElementsByClassName("collapse")){
      if (ele.classList.contains("in")) ele.classList.remove("in");
    }
  }
  else {
    instructionsCollapsed = false;
    for (let ele of document.getElementsByClassName("collapse")){
      if (ele.classList.contains("in")) continue;
      else ele.classList.add("in");
    }
  }
  recordClick("toggle-instructions");
}

// Auto shrink instructions text to fit in the window
let font_size = 11;
let heading_size = 13;
function dynamicInstructionsSize() {
  while (document.getElementById("heading").offsetHeight > (window.innerHeight * 0.9)) {
    console.log("decrease font size");
    font_size -= 1;
    heading_size -= 1;
    Array.from(document.getElementsByClassName("instructions-section")).forEach( ele => ele.style.fontSize = font_size + "pt" );
    Array.from(document.getElementsByClassName("instruction-headings")).forEach( ele => ele.style.fontSize = heading_size + "pt" );
  }
  if (!instructionsCollapsed) {
    while (document.getElementById("heading").offsetHeight < (window.innerHeight * 0.8)) {
      console.log("increase font size");
      font_size += 1;
      heading_size += 1;
      Array.from(document.getElementsByClassName("instructions-section")).forEach( ele => ele.style.fontSize = font_size + "pt" );
      Array.from(document.getElementsByClassName("instruction-headings")).forEach( ele => ele.style.fontSize = heading_size + "pt" );
    }
  }
}
dynamicInstructionsSize(); // Call once on page load
window.addEventListener('resize', dynamicInstructionsSize);  // And on resize

var submit_btn = document.getElementsByClassName("btn-default")[0];
submit_btn.classList.add("hidden");  // Hide the submit button to start

function checkSubmitDisplay() {
  //Only display the submit button if the worker has interacted with the dashboard and completed the survey
  if (ratingComplete && feedbackGiven && blockMarked && annotationFinished) {
    submit_btn.classList.remove("hidden");
    window.scrollTo(0,document.body.scrollHeight);
  }
}

function usabilityRated() {
  ratingComplete = true;
  recordClick("usability-rated");
  checkSubmitDisplay();
}

function giveFeedback() {
  feedbackGiven = true;
  recordClick("feedback");
  checkSubmitDisplay();
}