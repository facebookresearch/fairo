/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the logical form, surface form pair for move templates.
 */

import generateCodeArrayForTemplate from "../../helperFunctions/generateCodeArrayForTemplate";
import getSpans from "../../helperFunctions/getSpans";
import {getSurfaceForms} from "../../helperFunctions/getSurfaceForms";
import generateLocation from "./subTemplateGenerators/generateLocation";
import generateRepeatAction from "./subTemplateGenerators/generateRepeatAction";
import generateStopCondition from "./subTemplateGenerators/generateStopCondition";
var nestedProperty = require("nested-property");

function checkAndGenerateMoveTemplates(allBlocks,surfaceForms=undefined) {
  if (!surfaceForms) {
    surfaceForms = getSurfaceForms(allBlocks);
  }
  var code = generateCodeArrayForTemplate(allBlocks);
  var spans = getSpans(surfaceForms);
  console.log(spans);
  var text = generateMoveTemplates(allBlocks, surfaceForms, code, spans);
  
  // placeholder: refactoring code will go here
  
  document.getElementById("surfaceForms").innerText += text[0] + "\n";
  return text[0] + "    " + text[1];
}

function generateMoveTemplates(allBlocks, surfaceForms, code, spans) {
  var skeletal = {
    action: {
      action_type: "MOVE",
    },
  };
  var location = generateLocation(allBlocks, surfaceForms, code, spans);
  if (location) {
    nestedProperty.set(skeletal, "action.location", location);
  }
  var stopCondition = generateStopCondition(allBlocks, surfaceForms, code, spans);
  if (stopCondition) {
    nestedProperty.set(skeletal, "action.stop_condition", stopCondition);
  }

  var repeat = generateRepeatAction(allBlocks, surfaceForms, code, spans);
  if (repeat) {
    nestedProperty.set(skeletal, "action.repeat", repeat);
  }

  surfaceForms = surfaceForms.join(" ");

  return [surfaceForms, JSON.stringify(skeletal, null, 2)];
}

export default checkAndGenerateMoveTemplates;
