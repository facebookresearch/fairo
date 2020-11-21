/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the repeat
 * part of the action dictionary associated with a template. 
 * associated with a subtemplate
 */

import getIndicesForSpan from "../../../helperFunctions/getIndicesForSpan";

var nestedProperty = require("nested-property");

function generateRepeatAction(allBlocks,surfaceForms,code,spans){
    var repeat={};
    for(let i=0;i<allBlocks.length;i++){
      var block=allBlocks[i];
      var surfFormCurrent=surfaceForms[i];
     
      if (code[i]?.repeat?.repeat_count){
        var countSpan = getIndicesForSpan(surfaceForms, i, spans[i]);
        nestedProperty.set(repeat,"repeat_count",  [0, countSpan])   
    
      }
      if (code[i]?.repeat?.repeat_key){
        var countSpan = getIndicesForSpan(surfaceForms, i, spans[i]);
        nestedProperty.set(repeat,"repeat_key",  code[i]["action"]["repeat"]["repeat_key"])   
    
      }
      if (code[i]?.repeat?.repeat_dir){
        var countSpan = getIndicesForSpan(surfaceForms, i, spans[i]);
        nestedProperty.set(repeat,"repeat_dir", spans[i].toUpperCase())   
    
      }
    }
    if(JSON.stringify(repeat)==JSON.stringify({})) return null;
    return repeat;
  }
  export default generateRepeatAction;
