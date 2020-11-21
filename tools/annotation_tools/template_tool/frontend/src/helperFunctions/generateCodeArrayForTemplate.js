/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return an array of the codes
 *  associated with each template object of a template.
 */

var nestedProperty = require('nested-property');

/**
 *
 * @param {list} blocks The template to return logical forms for
 * @return {list} an array of logical forms
 */
function generateCodeArrayForTemplate(blocks) {
    const codeList = [];
    let templates = localStorage.getItem('templates');

    if (templates) {
        // template information exists
        templates = JSON.parse(templates);
    } else {
        // no template info exists
        templates = {};
    }

    blocks.forEach((element) => {
        const curCode = templates[element.getFieldValue('name')]['code'];
        let finalCode = curCode;
        let parent = '';
        const parentBlockConnection = element.parentBlock_;

        if (parentBlockConnection) {
            parent = parentBlockConnection.getFieldValue('parent');
        }

        if (parent) {
            finalCode = {};
            nestedProperty.set(finalCode, parent, curCode);
        }
        codeList.push(finalCode);
    });

    return codeList;
}
export default generateCodeArrayForTemplate;
