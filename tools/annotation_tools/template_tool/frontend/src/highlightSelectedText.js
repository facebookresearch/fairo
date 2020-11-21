/* Copyright (c) Facebook, Inc. and its affiliates. */

/** @fileoverview This file exports a function to enable
 * text highlighting. This allows the user to mark span
 * values in surface forms. The text highlighter also stores
 * information about the highlighted text to local storage*/

/**
 * This function allows highlighting text and
 *  saves information about highlighted text to local storage under "spans"
 */
function highlightSelectedText() {
  const selectedText = window.getSelection();

  // the text to be highlighted will be wrapped in a spam
  const span = document.createElement('span');

  const color = '#ff0000';

  // change the background color and weight of the created span
  span.style.backgroundColor = color;
  span.style.fontWeight = 'bold';

  const range = selectedText.getRangeAt(0).cloneRange();

  // surround the selected text with the span
  range.surroundContents(span);
  selectedText.removeAllRanges();
  selectedText.addRange(range);

  /*
   Save information about highlighted text under "spans".
   The highlighted part of the text is the span.
  */

  let text = document.getElementById('surfaceForms').innerText.split('\n');
  text = text[text.length - 1];
  if (!window.spans) {
    window.spans = {};
  }
  window.spans[text] = window.getSelection().toString();

  localStorage.setItem('spans', JSON.stringify(window.spans));
}

export default highlightSelectedText;
