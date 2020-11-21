/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the callback function for the "tagBlock"
 * option that the custom block has.
 */

import * as Blockly from 'blockly/core';
import saveToFile from '../../fileHandlers/saveToFile';

const tagBlockCallback = (block) => {
  console.log(block);
  const blockAsText = Blockly.Xml.domToText(
      Blockly.Xml.blockToDom(block, true),
  );
  const fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
  const tag = window.prompt('Please enter a tag.');
  let currentTagInfo = localStorage.getItem('tags');
  if (currentTagInfo) {
    currentTagInfo = JSON.parse(currentTagInfo);
  } else {
    currentTagInfo = {};
  }
  let infoOfCurTag = currentTagInfo[tag];
  if (!infoOfCurTag) {
    infoOfCurTag = [];
  }
  infoOfCurTag.push(fullBlockXML);
  currentTagInfo[tag] = infoOfCurTag;
  localStorage.setItem('tags', JSON.stringify(currentTagInfo));
  const currentDropdownInfo = JSON.parse(localStorage.getItem('blocks'));
  if (!currentDropdownInfo.includes(tag)) {
    currentDropdownInfo.push(tag);
  }
  localStorage.setItem('blocks', JSON.stringify(currentDropdownInfo));

  saveToFile();
  window.location.reload(true);
};

export default tagBlockCallback;
