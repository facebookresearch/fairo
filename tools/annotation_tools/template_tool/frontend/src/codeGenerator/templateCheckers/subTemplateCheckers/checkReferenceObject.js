/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the reference
 * object part of the action dictionary associated with a template.
 * associated with a subtemplate
 */

import getCountForSpan from "../../../helperFunctions/getCountForSpan";
import checkFilters from "./checkFilters";

var nestedProperty = require("nested-property");

function checkReferenceObject(allBlocks, surfaceForms, code, spans) {
  var referenceObject = {};
  for (let i = 0; i < allBlocks.length; i++) {
    var block = allBlocks[i];
    var surfFormCurrent = surfaceForms[i];
    if (code[i]?.reference_object?.special_reference) {
      referenceObject["special_reference"] =
        code[i]["reference_object"]["special_reference"];
    }
    if (code[i]?.reference_object?.repeat?.repeat_count) {
      var countSpan = getCountForSpan(surfaceForms, i, spans[i]);
      nestedProperty.set(referenceObject, "repeat.repeat_count", [
        0,
        countSpan,
      ]);
    }
    if (code[i]?.reference_object?.repeat?.repeat_key) {
      var countSpan = getCountForSpan(surfaceForms, i, spans[i]);
      nestedProperty.set(
        referenceObject,
        "repeat.repeat_key",
        code[i]["repeat"]["repeat_key"]
      );
    }
    if (code[i]?.reference_object?.repeat?.repeat_dir) {
      var countSpan = getCountForSpan(surfaceForms, i, spans[i]);
      nestedProperty.set(
        referenceObject,
        "repeat.repeat_dir",
        spans[i].toUpperCase()
      );
    }
  }

  // check filters
  var filters = checkFilters(allBlocks, surfaceForms, code, spans);
  if (filters) referenceObject["filters"] = filters;

  if (JSON.stringify(referenceObject) == JSON.stringify({})) return null;
  return referenceObject;
}

export default checkReferenceObject;
