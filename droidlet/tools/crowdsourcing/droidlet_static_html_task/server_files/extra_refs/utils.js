/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * utils.js contains a variety of click handlers and timing analytics for craftassist_task
 */

// Remove the stylesheet inherited from Mephisto by default
$('link[rel=stylesheet][href~="https://cdn.jsdelivr.net/npm/bulma@0.8.2/css/bulma.min.css"]').remove();

var timerStopped = false;
var ratingComplete = false;
var selfRatingComplete = false;
var clickedElements = new Array();

function recordClick(ele) {
  if (ele === "timerOFF") {  // Check to allow submission if all qualifications are met
    timerStopped = true;
    checkSubmitDisplay();
  }
  clickedElements.push({id: ele, timestamp: Date.now()});
  document.getElementById("clickedElements").value = JSON.stringify(clickedElements);
  console.log("Clicked elements array: " + JSON.stringify(clickedElements));
}
recordClick("start");

document.getElementsByClassName("btn-default")[0].classList.add("hidden");
function checkSubmitDisplay() {
  //Only display the submit button if the worker has interacted with the dashboard and completed the survey
  if (ratingComplete && selfRatingComplete && timerStopped) {
    document.getElementsByClassName("btn-default")[0].classList.remove("hidden");
    window.scrollTo(0,document.body.scrollHeight);
  }
}

function usabilityRated() {
  ratingComplete = true;
  recordClick("usability-rated");
  checkSubmitDisplay();
}

function selfRated() {
  selfRatingComplete = true;
  recordClick("self-rated");
  checkSubmitDisplay();
}

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
const numPages = 6;

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