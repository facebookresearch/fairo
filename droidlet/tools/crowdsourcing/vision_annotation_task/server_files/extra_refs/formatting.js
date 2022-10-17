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
        // document.getElementById("demos").style.marginTop = "0px";
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
    while (document.getElementById("heading").clientHeight > (window.innerHeight * 0.95)) {
        console.log("decrease font size");
        font_size -= 1;
        heading_size -= 1;
        Array.from(document.getElementsByClassName("instructions-section")).forEach( ele => ele.style.fontSize = font_size + "pt" );
        Array.from(document.getElementsByClassName("instruction-headings")).forEach( ele => ele.style.fontSize = heading_size + "pt" );
    }
    if (!instructionsCollapsed) {
        while (document.getElementById("heading").clientHeight < (window.innerHeight * 0.85)) {
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