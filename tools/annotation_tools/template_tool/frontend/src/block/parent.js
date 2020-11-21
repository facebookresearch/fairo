/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the definition and code
 * generator for the custom block used by the template generator.
 */

import * as Blockly from 'blockly/core';
import 'blockly/javascript';
import customInit from './customInit';
/**
 *
 * A parent block has a next connection, and a parent field.
 * The parent is the name of the spec-doc element that is being represented
 * as a parent.
 */

Blockly.Blocks['parent'] = {
  init: function() {
    this.appendValueInput('next')
        .setCheck(null)
        .appendField(new Blockly.FieldTextInput('Parent'), 'parent');

    this.setInputsInline(false);
    this.setColour(270);
    this.setTooltip('');
    this.setHelpUrl('');
    this.setOutput(true, null);

    customInit(this);
  },
};

Blockly.JavaScript['parent'] = function(block) {
  const code = block.getFieldValue('parent');
  return [JSON.stringify(code), Blockly.JavaScript.ORDER_ATOMIC];
};
