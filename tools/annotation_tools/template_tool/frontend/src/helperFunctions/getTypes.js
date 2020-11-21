/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return
 * an array of the types associated with a list of template objects.
 */

/**
 * This function returns an array of the names of the names
 * of an array of template objects.
 * @param {array} blocks
 * @return {array} An array of string names
 */
function getTypes(blocks) {
  const typeList = [];

  blocks.forEach((element) => {
    // push the type/name of this template object
    typeList.push(element.getFieldValue('name'));
  });

  return typeList;
}

export default getTypes;
