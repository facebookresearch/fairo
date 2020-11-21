/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines the filter function
 * for the block search dropdown.
 */

function filterFunction() {
  const input = document.getElementById('searchInput');
  const filter = input.innerText.toUpperCase();
  const listContainer = document.getElementById('UL');
  const listElements = listContainer.getElementsByTagName('li');
  let txtValue;


  /*
   Loop through all list items, and hide those who
   don't match the search query
  */

  for (let i = 0; i < listElements.length; i++) {
    const textContainer = listElements[i].getElementsByTagName('a')[0];
    txtValue = textContainer.textContent || textContainer.innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      listElements[i].style.display = '';
    } else {
      listElements[i].style.display = 'none';
    }
  }
}

export default filterFunction;
