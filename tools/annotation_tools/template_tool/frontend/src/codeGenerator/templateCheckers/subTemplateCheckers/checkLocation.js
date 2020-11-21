/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the location
 * part of the action dictionary associated with a template. 
 * associated with a subtemplate
 */

import getCountForSpan from "../../../helperFunctions/getCountForSpan";
import checkReferenceObject from "./checkReferenceObject"

var nestedProperty = require("nested-property");

function checkLocation(allBlocks,surfaceForms,code,spans){
    var location={};
  
    for(let i=0;i<allBlocks.length;i++){
      var block=allBlocks[i];
      var surfFormCurrent=surfaceForms[i];
      if (code[i]?.location?.steps) {
        var stepsSpan = getCountForSpan(surfaceForms, i, spans[i]);
        nestedProperty.set(location,"steps",[0,stepsSpan]);
      }
      if (code[i]?.location?.relative_direction) {
        nestedProperty.set(location,"relative_direction",spans[i].toUpperCase());
      } 
  
      if (code[i]?.location?.has_measure) {
        var measureSpan = getCountForSpan(surfaceForms, i, spans[i]);
        nestedProperty.set(location,"has_measure",[0,measureSpan]);
      } 
  
     if (code[i]?.location?.contains_coreference) {
        nestedProperty.set(location,"contains_coreference",true);
      }
    }
    var referenceObject=checkReferenceObject(allBlocks,surfaceForms,code,spans);
    if(referenceObject){
      location["reference_object"]=referenceObject;
    }
    if(JSON.stringify(location)==JSON.stringify({})) return null;
    return location;
  }
export default checkLocation 
