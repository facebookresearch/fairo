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

import getIndicesForSpan from "../../../helperFunctions/getIndicesForSpan";
var nestedProperty = require("nested-property");

function generateFilters(allBlocks, surfaceForms, code, spans) {
  var filters = {};
  for (let i = 0; i < allBlocks.length; i++) {
    var span=getIndicesForSpan(surfaceForms, i, spans[i]);
    // loop through all TOs
    
    if (code[i]?.filters?.has_colour) {
      nestedProperty.set(filters, "has_colour", [0, span]);
    }
    if (code[i]?.filters?.has_name) {
      nestedProperty.set(filters, "has_name", [0, span]);
    }
    if (code[i]?.filters?.has_size) {
      nestedProperty.set(filters, "has_size", [0, span]);
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

export default generateFilters;