/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines functions to return arrays
 * of surface forms given templates and an array of arrays of
 * surface forms.
 */

/**
 *
 * This is a function to get all surface forms associated with
 * all elements of an array of template objects. So, this function takes in
 * an array of blocks (template objects) and returns another array
 * containing the surface forms associated with each of them.
 * @param {array} allBlocks The blocks comprising the template
 * @return {array} An array of surface forms
 */
function generateAllSurfaceFormsForTemplate(allBlocks) {
  let templates = localStorage.getItem('templates');
  if (templates) {
    // template information exists
    templates = JSON.parse(templates);
  } else {
    // no templates saved
    templates = {};
  }

  const surfaceForms = [];

  allBlocks.forEach((element) => {
    // get surface form array for this element
    const surfaceForm =
      templates[element.getFieldValue('name')]['surfaceForms'];
    surfaceForms.push(surfaceForm);
  });

  return surfaceForms;
}

export { generateAllSurfaceFormsForTemplate };
