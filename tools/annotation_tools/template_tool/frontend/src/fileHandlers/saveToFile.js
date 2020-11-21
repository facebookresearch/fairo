/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the definition of a
 * function to dump the information in local storage to a file,
 * by making a request to the backend.
 */

function saveToFile() {
  // dump the current local storage information to a file
  const toSave = {};
  const spans = localStorage.getItem('spans');
  const savedBlocks = localStorage.getItem('savedByName');
  const savedByTag = localStorage.getItem('tags');
  const templates = localStorage.getItem('templates');
  const blocksInDropdown = localStorage.getItem('blocks');

  if (savedBlocks) {
    toSave['savedBlocks'] = JSON.parse(savedBlocks);
  }

  if (spans) {
    toSave['spans'] = JSON.parse(spans);
  }
  if (savedByTag) {
    toSave['savedByTag'] = JSON.parse(savedByTag);
  }
  if (templates) {
    toSave['templates'] = JSON.parse(templates);
  }
  if (blocksInDropdown) {
    toSave['blocks'] = JSON.parse(blocksInDropdown);
  }
  // request the backend to save this information
  callAPI(toSave);
}

function callAPI(data) {
  const HOST = 'http://localhost:';
  const PORT = '9000';
  fetch(HOST + PORT + '/readAndSaveToFile', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },

    body: JSON.stringify(data),
  })
    .then((res) => res.text())
    .then((res) => this.setState({ apiResponse: res }))
    .catch((err) => err);
}

export default saveToFile;
