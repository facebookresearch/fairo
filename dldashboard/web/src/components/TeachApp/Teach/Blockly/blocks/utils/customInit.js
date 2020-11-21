/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Function to provide different default block behavior. Used in block
 * initialization. See ../controls/controls_doUntil.js for an example.
 */

import Blockly from "blockly";

// is the block with the given id a descendant of the given ancestor?
const isDescendant = (potentialDescendantID, ancestor) => {
  return ancestor
    .getDescendants()
    .filter((d) => d.id !== ancestor.id) // remove self from descendants
    .map((d) => d.id)
    .includes(potentialDescendantID);
};

const customInit = (block) => {
  block.label = null;

  const removeLabelIfChanged = (event) => {
    const wasEventTypeChange =
      event.type === Blockly.Events.CREATE ||
      event.type === Blockly.Events.CHANGE;
    const wasChangeOnThisBlock = event.blockId === block.id;
    const wasFieldChanged =
      wasEventTypeChange &&
      (event.element === "field" || event.element === "inline") &&
      event.name !== Blockly.Block.COLLAPSED_FIELD_NAME;
    const wasChangeInAChildOfThisBlock = isDescendant(event.blockId, block);
    const wasFieldChangedInThisOrChild =
      (wasChangeOnThisBlock || wasChangeInAChildOfThisBlock) && wasFieldChanged;
    const wasChildBlockMoved =
      event.type === Blockly.Events.MOVE &&
      event.oldParentId !== event.newParentId &&
      wasChangeInAChildOfThisBlock;
    if (wasFieldChangedInThisOrChild || wasChildBlockMoved) {
      block.label = null;
    }
  };

  // a modal is a pop-up overlay within a browser tab, as part of the application
  // it does not open a new browser tab or window
  // https://material-ui.com/components/modal/
  const openModalCallback = () => {
    if (window.openSetLabelModal_) {
      window.openSetLabelModal_(block);
    } else {
      console.error("Could not find modal opening function.");
    }
  };

  const saveBlockCallback = () => {
    const blockAsText = Blockly.Xml.domToText(Blockly.Xml.blockToDom(block));
    const fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
    window.saveBlockToDatabase_(fullBlockXML, block.label);
  };

  const expandAllCallback = () => {
    block.setCollapsed(false);
    block.getDescendants().forEach((d) => d.setCollapsed(false));
  };

  // overwrite the default expand function with one that asks the user
  //   to add a label if they are going to collapse the block.
  const customExpandCallback = (value) => {
    if (block.label || block.collapsed_) {
      block.setCollapsed(value);
    } else {
      window.openSetLabelModal_(
        block,
        "Please add a label before collapsing the block."
      );
    }
  };

  // customizes the context menu on each block that customInit is called on
  const menuCustomizer = (menu) => {
    const commentIndex = menu.findIndex((o) => o.text === "Add Comment");
    menu.splice(commentIndex, 1); // remove default commenting features

    const collapseIndex = menu.findIndex(
      (o) => o.text === "Collapse Block" || o.text === "Expand Block"
    );
    menu.splice(collapseIndex, 1); // remove default collapsing features

    const expandAllOption = {
      text: "Expand All",
      enabled: !!block.collapsed_,
      callback: expandAllCallback,
    };
    menu.unshift(expandAllOption);

    const newCollapseOption = {
      text: (block.collapsed_ ? "Expand" : "Collapse") + " Block",
      enabled: true,
      callback: () => customExpandCallback(!block.collapsed_),
    };
    menu.unshift(newCollapseOption);

    const labelOption = {
      text: "Add Label",
      enabled: true,
      callback: openModalCallback,
    };
    menu.unshift(labelOption);

    const saveOption = {
      text: "Save Block",
      enabled: !!block.label,
      callback: saveBlockCallback,
    };
    menu.push(saveOption);

    return menu;
  };

  block.customContextMenu = menuCustomizer;

  block.setOnChange(removeLabelIfChanged);
};

export default customInit;
