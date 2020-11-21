/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a search function
 *  that places the block searched for by the user in
 * the dropdown into the toolbox.
 */

import Blockly from 'blockly/core';
import $ from 'jquery';
function searchForBlocks() {
  // add a default block to toolbox
  // document.getElementById("toolBox").innerHTML = ``;
  Blockly.mainWorkspace.updateToolbox(document.getElementById('toolBox'));

  // name/tag the user wants
  const nameOrTag = document.getElementById('searchInput').innerText;
  if (nameOrTag == 'Custom block') {
    document.getElementById(
        'toolBox',
    ).innerHTML = `<block xmlns="https://developers.google.com/blockly/xml" type="random"></block>
    <block xmlns="https://developers.google.com/blockly/xml" type="parent"></block>
    <block xmlns="https://developers.google.com/blockly/xml" type="textBlock"></block>`;
    Blockly.mainWorkspace.updateToolbox(document.getElementById('toolBox'));

    return;
  }

  const taggedInfoString = localStorage.getItem('tags');
  let taggedInfo;

  if (taggedInfoString) {
    // search in tags
    taggedInfo = JSON.parse(taggedInfoString);
    if (taggedInfo[nameOrTag]) {
      // this tag exists
      const blocks = taggedInfo[nameOrTag];
      blocks.forEach((element) => {
        const blockDom = Blockly.Xml.textToDom(element);
        console.log(blockDom);
        const blockInfo = blockDom.firstChild;

        // append the block to the toolbox and update it
        $('#toolBox').append(blockInfo);
        Blockly.mainWorkspace.updateToolbox(document.getElementById('toolBox'));
      });
    }
  }
  const namedInfoString = localStorage.getItem('savedByName');
  let namedInfo;
  if (namedInfoString) {
    // search in names
    namedInfo = JSON.parse(namedInfoString);
    if (namedInfo[nameOrTag]) {
      // this name exists

      const block = namedInfo[nameOrTag];
      const blockDom = Blockly.Xml.textToDom(block);
      const blockInfo = blockDom.firstChild;
      Blockly.Xml.domToWorkspace(blockDom, Blockly.mainWorkspace);
      //$('#toolBox').append(blockInfo);
      //Blockly.mainWorkspace.updateToolbox(document.getElementById('toolBox'));
    }
  }
}

export default searchForBlocks;
