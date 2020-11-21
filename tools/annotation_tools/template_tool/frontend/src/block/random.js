/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the definition and code generator for the
 * "random generator" block used by the template generator.
 */

import * as Blockly from 'blockly/core';
import 'blockly/javascript';
import customInit from './customInit';

Blockly.Blocks['random'] = {
  init: function () {
    this.appendValueInput('next')
      .setCheck(null)
      .appendField(new Blockly.FieldTextInput('name'), 'name')
      .appendField('-')
      .appendField('random over')
      .appendField(new Blockly.FieldTextInput('default'), 'randomCategories');
    this.setOutput(true, null);
    this.setColour(230);
    this.setTooltip('');
    this.setHelpUrl('');
    customInit(this);
  },
};

Blockly.JavaScript['random'] = function (block) {
  console.log(block.getNextStatement());
  // Blockly.JavaScript.statementToCode(block,"next");
  const valueName = block.getFieldValue('name');

  // Information about template/template objects
  const templatesString = localStorage.getItem('templates');

  let code;

  if (templatesString) {
    // If information about the template/template object exists, we use it
    const templates = JSON.parse(templatesString);
    code = templates[valueName]['code'];
  }

  return [JSON.stringify(code), Blockly.JavaScript.ORDER_ATOMIC];

  // return [JSON.stringify(code), Blockly.JavaScript.ORDER_ATOMIC];
};
