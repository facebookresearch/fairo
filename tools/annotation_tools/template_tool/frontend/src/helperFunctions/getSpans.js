/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the spans
 * associated with surface forms
 */

/**
 *
 * @param {array} surfaceForms The surface forms to provide spans for
 * @return {array} an array of the associated spans
 */
function getSpans(surfaceForms) {
  console.log(surfaceForms);
  const spans = [];
  for (let i = 0; i < surfaceForms.length; i++) {
    const surfaceForm = surfaceForms[i];
    let span = window.spans[surfaceForm];
    if (!span) {
      // the entire surface form is the span
      span = surfaceForm;
    }
    spans.push(span);
  }
  console.log(spans);
  return spans;
}
export default getSpans;
