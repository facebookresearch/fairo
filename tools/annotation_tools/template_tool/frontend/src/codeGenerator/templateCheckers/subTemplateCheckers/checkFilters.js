/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the filters
 * part of the action dictionary associated with a template.
 * associated with a subtemplate
 */

import getCountForSpan from "../../../helperFunctions/getCountForSpan";
var nestedProperty = require("nested-property");

function checkFilters(allBlocks, surfaceForms, code, spans) {
  var filters = {};
  for (let i = 0; i < allBlocks.length; i++) {
    // loop through all TOs

    var surfFormCurrent = surfaceForms[i];
    if (code[i]?.filters?.has_colour) {
      var colourSpan = getCountForSpan(surfaceForms, i, spans[i]);
      nestedProperty.set(filters, "has_colour", [0, colourSpan]);
    }
    if (code[i]?.filters?.has_name) {
      var nameSpan = getCountForSpan(surfaceForms, i, spans[i]);
      nestedProperty.set(filters, "has_name", [0, nameSpan]);
    }
    if (code[i]?.filters?.has_size) {
      var sizeSpan = getCountForSpan(surfaceForms, i, spans[i]);
      nestedProperty.set(filters, "has_size", [0, sizeSpan]);
    }
    if (code[i]?.filters?.contains_coreference) {
      nestedProperty.set(filters, "contains_coreference", true);
    }
    if (code[i]?.filters?.author) {
      nestedProperty.set(filters, "author", code[i]["filters"]["author"]);
    }
  }

  // Placeholder: check argmax and argmin

  if (JSON.stringify(filters) == JSON.stringify({})) return null;

  return filters;
}

export default checkFilters;
