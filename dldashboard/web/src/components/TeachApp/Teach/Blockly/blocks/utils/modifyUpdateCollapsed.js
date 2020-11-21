/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This file, when imported, updates the Blockly default collapse behavior
 * to display a custom label (when it's present) on collapsed blocks.
 * The custom field name used is <block>.label
 *
 * Original code is here:
 * https://github.com/google/blockly/blob/9e98df9949292ee114ab159289bc5cf58a4b7b45/core/block_svg.js
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";

// Save old collapse function to use when no label is present
const oldUpdateCollapsed = Blockly.BlockSvg.prototype.updateCollapsed_;
Blockly.BlockSvg.prototype.oldUpdateCollapsed_ = oldUpdateCollapsed;

Blockly.BlockSvg.prototype.updateCollapsed_ = function () {
  if (this.label) {
    var collapsed = this.isCollapsed();
    var collapsedInputName = Blockly.Block.COLLAPSED_INPUT_NAME;
    var collapsedFieldName = Blockly.Block.COLLAPSED_FIELD_NAME;

    for (let i = 0, input; (input = this.inputList[i]); i++) {
      if (input.name !== collapsedInputName) {
        input.setVisible(!collapsed);
      }
    }

    if (!collapsed) {
      this.removeInput(collapsedInputName);
      return;
    }

    var icons = this.getIcons();
    for (let i = 0, icon; (icon = icons[i]); i++) {
      icon.setVisible(false);
    }

    var text = this.label;
    var field = this.getField(collapsedFieldName);
    if (field) {
      field.setValue(text);
      return;
    }
    var input =
      this.getInput(collapsedInputName) ||
      this.appendDummyInput(collapsedInputName);
    input.appendField(new Blockly.FieldLabel(text), collapsedFieldName);
  } else {
    this.oldUpdateCollapsed_();
  }
};
