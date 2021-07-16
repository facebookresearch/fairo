/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Registers a Blockly mutator that serializes the label for each block
 * into and out of the database. For more info on mutators:
 * https://developers.google.com/blockly/guides/create-custom-blocks/extensions
 */

import Blockly from "blockly";

const labelMutator = {};

labelMutator.mutationToDom = function () {
  const container = document.createElement("mutation");
  const label = this.label;
  if (label) {
    container.setAttribute("label", label);
  }
  return container;
};

labelMutator.domToMutation = function (xmlElement) {
  var label = xmlElement.getAttribute("label");
  if (label) {
    this.label = label;
  }
};

Blockly.Extensions.registerMutator("labelMutator", labelMutator);
