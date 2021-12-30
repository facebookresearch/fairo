/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * utils.js contains a variety of click handlers and timing analytics for vision_annotation_task
 */

// Remove the stylesheet inherited from Mephisto by default
$('link[rel=stylesheet][href~="https://cdn.jsdelivr.net/npm/bulma@0.8.2/css/bulma.min.css"]').remove();

var clickedElements = new Array();
var complete = [false, false, false, false]; 
var correct = [false, false, false, false];
var answers = [
  [[-2,0,2], [-2,1,2], [-2,2,2]],
  [[-5,1,0],[-4,1,0],[-5,0,0],[-4,0,0],[-4,1,-1],[-5,1,-1],[-4,0,-1],[-5,0,-1]],
  null,
  [[0,4,0],[1,4,0],[0,4,1],[1,4,1]]
]

function checkAnswer(qnum, ans) {
  console.log("Checking Question #: " + qnum + " answer: " + ans);
  let qidx = qnum -1;
  complete[qidx] = true;
  if (ans === "null") {
    if (JSON.stringify(JSON.parse(ans)) === JSON.stringify(answers[qidx])){
      console.log("answer " + qnum + " correct");
      correct[qidx] = true;
    } else {
      correct[qidx] = false;
      console.log("answer " + qnum + " incorrect");
    }
  }
  else if (JSON.stringify(JSON.parse(ans).sort()) === JSON.stringify(answers[qidx].sort())){
    console.log("answer " + qnum + " correct");
    correct[qidx] = true;
  } else {
    correct[qidx] = false;
    console.log("answer " + qnum + " incorrect");
  }
  let qid = "q" + qnum + "Answer";
  document.getElementById(qid).value = JSON.stringify(correct[qidx]);
  if (complete.every(x => x === true)) {
    document.getElementById("complete-prompt").innerHTML = "";
  }
}

// recordClick records and handles messages received from within the iframes
function recordClick(ele) {
  if (ele["q1__output"]) { checkAnswer(1, ele.q1__output) }
  else if (ele["q2__output"]) { checkAnswer(2, ele.q2__output) }
  else if (ele["q3__output"]) { checkAnswer(3, ele.q3__output) }
  else if (ele["q4__output"]) { checkAnswer(4, ele.q4__output) }

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
    //console.log(data);
    if (data.msg) recordClick(data.msg);
  }, false);
}
else if (window.attachEvent) {  // Cross compatibility for old versions of IE
  window.attachEvent("onmessage", (event) => {
    if (typeof(event.data) === "string") {
      data = JSON.parse(event.data);
    } else data = event.data
    //console.log(data);
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

  // if at the bottom of the page and all questions are not done, prompt user
  if ((window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 10)) {
    if (!complete.every(x => x === true)) {
      document.getElementById("complete-prompt").innerHTML = "Not all questions are complete, are you sure you want to submit?";
    }
    else {
      document.getElementById("complete-prompt").innerHTML = "";
    }
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
