/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file returns the count for a span in a template 
 * i.e. [startIndex, endIndex], given a list of surface forms.
 */

function getCountForSpan(surfaceFormList, i, span) {
    var startSpan = 0;
    var endSpan = 0;
    var startSurfForm = 0;
    for (var j = 0; j < i; j++) {
      var surfaceForm = surfaceFormList[j].split(" ");
      startSurfForm += surfaceForm.length;
    }
  
    var surfFormToCheck = surfaceFormList[i].split(" ");
    var spanArray = span.split(" ");

    // add index of the span to the start
    var startSpan = startSurfForm + surfFormToCheck.indexOf(spanArray[0]);
  
    endSpan = spanArray.length + startSpan - 1;
    console.log(startSpan);
    console.log(endSpan);
    return [startSpan, endSpan];
  }
  
export default getCountForSpan;