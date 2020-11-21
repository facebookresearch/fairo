/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the callback function for the "saveBlock"
 * option that the custom block has. This saves information about the block to
 * local storage and the template library file.
 */

import saveTemplateObject from '../../saveToLocalStorage/saveTemplateObject';
import * as Blockly from 'blockly/core';
import saveTemplate from '../../saveToLocalStorage/saveTemplate';
import saveRandomTemplateObject from '../../saveToLocalStorage/saveRandom';
function saveBlockCallback(block) {
  const blockAsText = Blockly.Xml.domToText(
    Blockly.Xml.blockToDom(block, true),
  );

  // wrap the block in xml tags
  const fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
  let name = block.getFieldValue('name');
  const allBlocks = Blockly.mainWorkspace.getAllBlocks();
  if (allBlocks.length != 1) {
    // not a single template object
    name = window.prompt('Enter a name for the template');
  }

  // get the blocks currently saved by name
  const currentSavedInfoString = localStorage.getItem('savedByName');
  let currentSavedInfo;

  if (currentSavedInfoString) {
    // blocks have already been saved
    currentSavedInfo = JSON.parse(currentSavedInfoString);
  } else {
    // no blocks saved, initialise the dictionary.
    currentSavedInfo = {};
  }

  // save this block
  currentSavedInfo[name] = fullBlockXML;

  localStorage.setItem('savedByName', JSON.stringify(currentSavedInfo));

  const currentDropdownInfo = JSON.parse(localStorage.getItem('blocks'));
  if (!currentDropdownInfo.includes(name)) {
    currentDropdownInfo.push(name);
  }

  localStorage.setItem('blocks', JSON.stringify(currentDropdownInfo));

  // wrap the block name in option tags
  if (allBlocks.length == 1) {
    // it is a template object
    if (allBlocks[0].type == 'random') {
      saveRandomTemplateObject(block, name);
    } else {
      saveTemplateObject(block, name);
    }
  } else {
    saveTemplate(block, name);
  }

  // refresh the dropdown selections
  window.location.reload(true);

  Blockly.Xml.DomToWorkspace(
    Blockly.Xml.blockToDom(block, true),
    Blockly.mainWorkspace,
  );
  console.log(Blockly.mainWorkspace.getAllBlocks());
}

export default saveBlockCallback;
