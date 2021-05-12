/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Blockly/React interface components.
 */

import React, { useEffect, useRef } from "react";

import Blockly from "blockly/core";
import locale from "blockly/msg/en";

import "blockly/blocks";
import "./blocks";

export { default as Toolbox } from "./Toolbox";

Blockly.setLocale(locale);

export const Block = ({ children, ...props }) => {
  return (
    <block is="blockly" {...props}>
      {children}
    </block>
  );
};

export const Category = ({ children, name, ...props }) => {
  return (
    <category is="blockly" name={name} {...props}>
      {children}
    </category>
  );
};

// Component that injects a Blockly editor into itself.
// Children are the components of the workspace.
const BlocklyEditor = React.forwardRef(
  ({ initialXml, children, ...rest }, workspaceRef) => {
    const blocklyDiv = useRef();
    const toolbox = useRef();

    useEffect(() => {
      workspaceRef.current = Blockly.inject(blocklyDiv.current, {
        toolbox: toolbox.current,
        ...rest,
      });

      if (initialXml) {
        Blockly.Xml.domToWorkspace(
          Blockly.Xml.textToDom(initialXml),
          workspaceRef.current
        );
      }

      // HACK: need to ensure we don't recreate the workspace ever
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
      <>
        <div id="blocklyDiv" ref={blocklyDiv}></div>
        <xml
          xmlns="https://developers.google.com/blockly/xml"
          is="blockly"
          style={{ display: "none" }}
          ref={toolbox}
        >
          {children}
        </xml>
      </>
    );
  }
);

export default BlocklyEditor;
