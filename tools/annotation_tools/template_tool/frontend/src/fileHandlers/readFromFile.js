/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the definition of a function
 * to restore local storage by requesting the backend for the
 * previously dumped contents of local storage.
 */


/**
 *
 * This function restores local storage information by requesting
 * the backend to provide the dumped contents of local storage.
 */

function restore() {
  const HOST = 'http://localhost:';
  const PORT = '9000';
  fetch(HOST + PORT + '/readAndSaveToFile')
    .then((res) => res.text())
    .then((res) => {
      const result = JSON.parse(res);
      if (result['savedBlocks']) {
        localStorage.setItem('savedByName',
          JSON.stringify(result['savedBlocks']));
      }
      if (result['templates']) {
        localStorage.setItem('templates',
          JSON.stringify(result['templates']));
      }
      if (result['savedByTag']) {
        localStorage.setItem('tags', JSON.stringify(result['savedByTag']));
      }
      if (result['spans']) {
        localStorage.setItem('spans', JSON.stringify(result['spans']));
        window.spans = result['spans'];
      }
      if (result['blocks']) {
        localStorage.setItem('blocks', JSON.stringify(result['blocks']));
      }

      if (!localStorage.getItem('reload')) {
        /* set reload to true and then reload the page */
        localStorage.setItem('reload', 'true');
        window.location.reload();
      } else {
        /* after reloading remove "reload" from localStorage */
        localStorage.removeItem('reload');
      }
    });
}

export default restore;
