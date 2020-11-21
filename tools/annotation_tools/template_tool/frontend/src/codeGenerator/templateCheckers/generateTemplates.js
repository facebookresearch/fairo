/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the logical form, surface form pair for templates.
 */

import checkAndGenerateMoveTemplates from "./moveTemplates";

function generateTemplates(allBlocks,surfaceFormsPicked){
    // call the right template generator for the template
    // for now only move templates are supported
    return checkAndGenerateMoveTemplates(allBlocks,surfaceFormsPicked);
}
export default generateTemplates