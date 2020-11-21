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
import * as Blockly from 'blockly/core';
import saveToFile from '../../fileHandlers/saveToFile';

function getKeyByValue(object, value) {
  return Object.keys(object).find(key => object[key] === value);
}


function deleteBlockCallback(block) {
  const blockAsText = Blockly.Xml.domToText(
    Blockly.Xml.blockToDom(block, true),
  );

  // wrap the block in xml tags
  const fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
  let name = block.getFieldValue('name');
  const allBlocks = Blockly.mainWorkspace.getAllBlocks();
  if (allBlocks.length != 1) {
    // not a single template object, get name from dict
    let savedBlockDict = JSON.parse(localStorage.getItem('savedByName'));
    name = getKeyByValue(savedBlockDict,fullBlockXML);
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
  // remove this block from current info
  delete currentSavedInfo[name];
  localStorage.setItem('savedByName', JSON.stringify(currentSavedInfo));

  // Now fix blocks
  const currentDropdownInfo = JSON.parse(localStorage.getItem('blocks'));
  var blockIndex = currentDropdownInfo.indexOf(name);// get name index
  if (blockIndex != -1) {
    // name found, delete block now
    currentDropdownInfo.splice(blockIndex, 1);
  }
  localStorage.setItem('blocks', JSON.stringify(currentDropdownInfo));

  // Now fix templates
  const currentTemplatesString = localStorage.getItem('templates');
  let currentTemplates;
  if (currentTemplatesString) {
    // blocks have already been saved
    currentTemplates = JSON.parse(currentTemplatesString);
  } else {
    // no blocks saved, initialise the dictionary.
    currentTemplates = {};
  }
  delete currentTemplates[name];
  localStorage.setItem('templates', JSON.stringify(currentTemplates));

  // delete from file
  saveToFile();

  // refresh the dropdown selections
  window.location.reload(true);

  Blockly.Xml.DomToWorkspace(
    Blockly.Xml.blockToDom(block, true),
    Blockly.mainWorkspace,
  );
  console.log(Blockly.mainWorkspace.getAllBlocks());
}

export default deleteBlockCallback;
