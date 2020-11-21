/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Modal component allowing labels to be added to Blockly blocks.
 */

import React, { useEffect, useState } from "react";
import Blockly from "blockly";

import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";
import Dialog from "@material-ui/core/Dialog";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";
import DialogTitle from "@material-ui/core/DialogTitle";

export default function LabelBlockModal({
  block,
  open,
  closeModal,
  saveBlockToDatabase,
  error,
}) {
  const labelOrEmptyStr = (block && block.label) || "";
  const [label, setLabel] = useState("");

  useEffect(() => {
    setLabel(labelOrEmptyStr);
  }, [labelOrEmptyStr]);

  // Does tasks necessary when closing the modal and saving a block
  // including collapsing the block, refreshing cached label, closing modal itself
  const closeModalOnSaveHelper = () => {
    block.setCollapsed(false); // toggle off/on to refresh label
    block.setCollapsed(true);
    setLabel("");
    closeModal();
  };

  const handleSaveLabel = () => {
    // HACK this goes against functional React paradigm but is necessary to interface with Blockly
    block.label = label;
    closeModalOnSaveHelper();
  };

  const handleSaveLabelAndBlock = () => {
    // HACK this goes against functional React paradigm but is necessary to interface with Blockly
    block.label = label;
    const blockAsText = Blockly.Xml.domToText(Blockly.Xml.blockToDom(block));
    const fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
    saveBlockToDatabase(fullBlockXML, label);
    closeModalOnSaveHelper();
  };

  return (
    <Dialog
      open={open}
      onClose={closeModal}
      aria-labelledby="form-dialog-title"
    >
      <DialogTitle id="form-dialog-title">Label Block</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Type a name for the block you right-clicked on in the box below. Once
          you add a label, you can save the label, or save the block to the
          database with the label you wrote.
        </DialogContentText>
        <DialogContentText color="error">{error}</DialogContentText>
        <TextField
          autoFocus
          margin="dense"
          id="name"
          label="Block Label"
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          fullWidth
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={closeModal} color="secondary">
          Cancel
        </Button>
        <Button onClick={handleSaveLabel} color="secondary" disabled={!label}>
          Save Label
        </Button>
        <Button
          onClick={handleSaveLabelAndBlock}
          color="secondary"
          disabled={!label}
        >
          Save Block
        </Button>
      </DialogActions>
    </Dialog>
  );
}
