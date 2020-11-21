/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to save a template
 * to local storage, and then call upon a function to dump
 * contents of local storage to a file.
 */

import * as Blockly from 'blockly/core';
import saveToFile from '../fileHandlers/saveToFile';
import generateCodeArrayForTemplate from
  '../helperFunctions/generateCodeArrayForTemplate';
import { generateAllSurfaceFormsForTemplate } from
  '../helperFunctions/getSurfaceForms';

/**
 * This function saves a template to local storage in a form
 * that can be processed by the Python generator.
 */
function saveTemplate(block, name) {
  //const allBlocks = Blockly.mainWorkspace.getAllBlocks();
  const allBlocksInWorkspace = Blockly.mainWorkspace.getAllBlocks();
  const allBlocks = [];
  for (let i = 0; i < allBlocksInWorkspace.length; i++) {
    if (
      allBlocksInWorkspace[i].type == 'textBlock' ||
      allBlocksInWorkspace[i].type == 'random'
    ) {
      allBlocks.push(allBlocksInWorkspace[i]);
    }
  }

  var templates = {};
  if (localStorage.getItem('templates')) {
    // some templates have been stored already
    templates = JSON.parse(localStorage.getItem('templates'));
  }
  templates[name] = { surfaceForms: '', code: '' };
  templates[name]['surfaceForms'] = generateAllSurfaceFormsForTemplate(
    allBlocks,
  );
  templates[name]['code'] = generateCodeArrayForTemplate(allBlocks);
  localStorage.setItem('templates', JSON.stringify(templates));
  saveToFile();
}
export default saveTemplate;
