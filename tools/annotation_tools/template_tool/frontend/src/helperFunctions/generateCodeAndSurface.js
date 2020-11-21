/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return an array of the
 * surface and logical forms associated with each template object of a template.
 */

/**
 * This function returns the logical and surface form array
 * associated with a list of blocks/ a template
*/
import * as Blockly from 'blockly/core';

export function getAllSurfaceFormsAndCode(block) {
  let curCode = null;
  let surfaceForms = [];
  let allBlocks = block;
  let templates = localStorage.getItem('templates');

  if (templates) {
    // template information exists
    templates = JSON.parse(templates);
  } else {
    // no template info exists
    templates = {};
  }

  // push code for this element
  // const parent=element.getFieldValue("parent");
  if (!templates[block.getFieldValue("name")]) {
    return [block, '', {}];
  }
  curCode = templates[block.getFieldValue('name')]['code'];
  surfaceForms =
    templates[block.getFieldValue('name')]['surfaceForms'];

  return [allBlocks, surfaceForms, curCode];
}


export function generateCodeAndSurfaceForm(blocks) {
  const codeList = [];
  const surfaceForms = [];
  const allBlocks = [];
  let templates = localStorage.getItem('templates');
  let savedBlocks = JSON.parse(localStorage.getItem('savedByName'));

  if (templates) {
    // template information exists
    templates = JSON.parse(templates);
  } else {
    // no template info exists
    templates = {};
  }

  
  blocks.forEach((element) => {
    // push code for this element
    // const parent=element.getFieldValue("parent");
    if (element.type != 'random') {
      if (!templates[element.getFieldValue("name")]) {
        return false;
      }
      const curCode = templates[element.getFieldValue('name')]['code'];
      codeList.push(curCode);
      let surfaceForm =
        templates[element.getFieldValue('name')]['surfaceForms'];
      surfaceForm = randomFromList(surfaceForm);
      surfaceForms.push(surfaceForm);
      allBlocks.push(element);
    } else {
      // this is random
      const randomChoices = element
        .getFieldValue('randomCategories')
        .split(', ');

      const choice = randomFromList(randomChoices);
      const block = savedBlocks[choice];
      const blockDom = Blockly.Xml.textToDom(block);
      const blockInfo = blockDom.firstChild;
      allBlocks.push(blockInfo);
    
      if (!templates[choice]) return false;
  
      const curCode = templates[choice]['code'];
      let surfaceForm = templates[choice]['surfaceForms'];
      // if template
      if (Array.isArray(curCode)) {
        for (let i = 0; i < surfaceForm.length; i++) {
            const s = randomFromList(surfaceForm[i]);
            surfaceForms.push(s);
            codeList.push(curCode[i]);
        }
      } else {
        codeList.push(curCode);
        surfaceForm = randomFromList(surfaceForm);
        surfaceForms.push(surfaceForm);
      }
    }
  });

  return [allBlocks, surfaceForms, codeList];
}

export default generateCodeAndSurfaceForm;

/**
 * This function returns a random element from a list
 */
function randomFromList(list) {
  const numberLength = list.length;
  const x = Math.floor(Math.random() * numberLength);
  return list[x];
}
