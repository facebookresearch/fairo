/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the stop 
 * condition part of the action dictionary associated with a template. 
 * associated with a subtemplate
 */

import getCountForSpan from "../../../helperFunctions/getCountForSpan";

var nestedProperty = require("nested-property");

function checkStopCondition(allBlocks,surfaceForms,code,spans){
    var stopCondition={};
    for(let i=0;i<allBlocks.length;i++){
      var block=allBlocks[i];
      var surfFormCurrent=surfaceForms[i];
      if(code[i]?.stop_condition?.condition_type){
        nestedProperty.set(stopCondition,"condition_type",code[i]["stop_condition"]["condition_type"]);
      }
      if(code[i]?.block_type){
        var countSpan = getCountForSpan(surfaceForms, i, spans[i]);
        nestedProperty.set(stopCondition,"block_type",countSpan);
      }
  
  
    }
    if(JSON.stringify(stopCondition)==JSON.stringify({})) return null;
    return stopCondition;
  }  
  
  export default checkStopCondition
