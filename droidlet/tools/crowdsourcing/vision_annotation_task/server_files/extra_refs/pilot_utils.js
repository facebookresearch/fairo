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
  [[-1,4,-1],[0,4,-1],[-1,4,0],[0,4,0]]
]

// Check submissions
function checkAnswer(qnum, ans) {
  try {
    parsedAnswer = JSON.parse(ans)[0]["locs"];
  } catch {
    parsedAnswer = JSON.parse(ans);
  }
  console.log("Checking Question #: " + qnum + " answer: " + JSON.stringify(parsedAnswer));
  let qidx = qnum -1;
  complete[qidx] = true;
  if (ans === "null" || answers[qidx] === "null") {  // can't sort null
    if (JSON.stringify(parsedAnswer) === JSON.stringify(answers[qidx])){
      correct[qidx] = true;
    } else {
      correct[qidx] = false;
    }
  }
  else if (JSON.stringify(parsedAnswer.sort()) === JSON.stringify(answers[qidx].sort())){
    correct[qidx] = true;
  } else {
    correct[qidx] = false;
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

  // Record everything, can be parsed later for timing analytics if desired
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
    if (data.msg) recordClick(data.msg);
  }, false);
}
else if (window.attachEvent) {  // Cross compatibility for old versions of IE
  window.attachEvent("onmessage", (event) => {
    if (typeof(event.data) === "string") {
      data = JSON.parse(event.data);
    } else data = event.data
    if (data.msg) recordClick(data.msg);
  });
}

// if at the bottom of the page and all questions are not done, prompt user
document.addEventListener('scroll', function(e) {
  if ((window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 10)) {
    if (!complete.every(x => x === true)) {
      document.getElementById("complete-prompt").innerHTML = "Not all questions are complete, are you sure you want to submit?";
    }
    else {
      document.getElementById("complete-prompt").innerHTML = "";
    }
  }
});

// Bug report onclick
function bugReport() {
  document.getElementById('bug').value = "true";
  document.getElementById('bugButton').disabled = true;
}
