/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to
 * save a random block to localStorage.
 */

import saveToFile from '../fileHandlers/saveToFile';

/**
 * This function saves a random block to local storage.
 * @param {block} block The block to be saved
 * @param {string} name The name of the block
 */
function saveRandomTemplateObject(block, name) {
  let templates = {};
  if (localStorage.getItem('templates')) {
    // some templates have been stored already
    templates = JSON.parse(localStorage.getItem('templates'));
  }
  const randomOver = block.getFieldValue('randomCategories').split(', ');
  const surfaceForms = [];
  const codes = [];
  randomOver.forEach((templateObject) => {
    const code = templates[templateObject]['code'];
    const surfaceForm = templates[templateObject]['surfaceForms'];
    codes.push(code);
    surfaceForms.push(surfaceForm);
  });
  templates[name] = {};
  templates[name]['code'] = codes;
  templates[name]['surfaceForms'] = surfaceForms;
  localStorage.setItem('templates', JSON.stringify(templates));
  saveToFile();
}

export default saveRandomTemplateObject;
