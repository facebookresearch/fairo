/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import * as Blockly from 'blockly/core';

import getTypes from './helperFunctions/getTypes';
import saveToFile from './fileHandlers/saveToFile';

/**
 * This function stores refactors in surface forms with the template
 */
function saveChanges() {
  const generated = localStorage.getItem('current').split('\n');
  const current = document.getElementById('surfaceForms').innerText.split('\n');
  const allBlocks = Blockly.mainWorkspace.getAllBlocks();
  const types = getTypes(allBlocks).join(' ');
  let savedInfo = localStorage.getItem('templates');
  if (savedInfo) {
    savedInfo = JSON.parse(savedInfo);
  } else {
    savedInfo = {};
  }
  if (!savedInfo[types]) {
    window.alert('save the template first');
    return;
  }
  savedInfo[types]['changes'] = [];
  for (let i = 0; i < generated.length; i++) {
    if (generated[i] != current[i]) {
      savedInfo[types]['changes'].push([generated[i], current[i]]);
    }
  }
  localStorage.setItem('templates', JSON.stringify(savedInfo));
  saveToFile();
}
export default saveChanges;
