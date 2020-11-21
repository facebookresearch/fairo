/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import Terminal from "react-console-emulator";

const commands = {
  echo: {
    description: "Echo a passed string.",
    usage: "echo <string>",
    fn: function () {
      return `${Array.from(arguments).join(" ")}`;
    },
  },
};

/**
 * A wrapper around the console emulator component.
 * Eventually can be connected to either an openable Python
 * debugger or to the terminal, based on need
 */
class Console extends React.Component {
  render() {
    return (
      <Terminal
        commands={commands}
        promptLabel={"me@locobot:~$ (not connected to anything right now)"}
      />
    );
  }
}

export default Console;
