/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Modal component allowing labels to be added to Blockly blocks.
 */

import React from "react";
import { CopyBlock, dracula } from "react-code-blocks";

import Button from "@material-ui/core/Button";
import Dialog from "@material-ui/core/Dialog";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogTitle from "@material-ui/core/DialogTitle";

export default function BasicModal({ open, close, text_to_render }) {
  return (
    <Dialog open={open} aria-labelledby="form-dialog-title">
      <DialogTitle id="form-dialog-title">Generated Code</DialogTitle>
      <DialogContent>
        <CopyBlock
          text={text_to_render}
          language="python"
          wrapLines
          theme={dracula}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={close} color="secondary">
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  );
}
