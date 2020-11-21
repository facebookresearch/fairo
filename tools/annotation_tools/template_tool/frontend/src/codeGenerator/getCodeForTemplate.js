/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the logical form, surface form pair for templates.
 */

import * as Blockly from "blockly/core";
import generateCodeArrayForTemplate from "../helperFunctions/generateCodeArrayForTemplate";
import getTypes from "../helperFunctions/getTypes";
import {getSurfaceFormsFromList,generateAllSurfaceFormsForTemplate} from "../helperFunctions/getSurfaceForms";
import generateTemplates from "./templateCheckers/generateTemplates";

function getCodeForTemplate() {
  var allBlocks = Blockly.mainWorkspace.getAllBlocks();
  var code = generateCodeArrayForTemplate(allBlocks);
  var savedTemplates = localStorage.getItem("templates");
  if (savedTemplates) {
    // these are templates/template objects that the user has saved to local storage and file.
    savedTemplates = JSON.parse(savedTemplates);
    var types = getTypes(allBlocks).join(" ");
    if (savedTemplates.hasOwnProperty(types)) {
      
      // get saved logical forms for each TO
      var code = savedTemplates[types]["code"];

      // get saved surface forms for each TO
      var surfaceForms = savedTemplates[types]["surfaceForms"];
      
      // randomly pick surface forms form saved surface forms
      var surfaceFormsPicked = getSurfaceFormsFromList(surfaceForms);
      
      // get the logical and surface form for the template
      var logicalAndSurfaceForm = generateTemplates(
        allBlocks,
        surfaceFormsPicked
      );

      document.getElementById("actionDict").innerText +=
        "\n" + logicalAndSurfaceForm;

      return;
    }
  }

  // get the logical and surface form for the template
  var code = generateTemplates(allBlocks);
  document.getElementById("actionDict").innerText += "\n" + code;
}

export default getCodeForTemplate;
