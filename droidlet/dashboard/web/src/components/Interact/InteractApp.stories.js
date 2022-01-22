/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LocoView.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import InteractApp from "./InteractApp";
import Message from "./Message";
import Question from "./Question";
import AgentThinking from "./AgentThinking";
import testStore from "../../TestStore";

export default {
  component: InteractApp,
  title: "InteractApp",
  // Our exports that end in "Data" are not stories.
  excludeStories: /.*Data$/,
};

export const Default = () => {
  return <InteractApp stateManager={testStore} />;
};

export const Chat = () => {
  return (
    <Message
      stateManager={testStore}
      chats={[{ msg: "Chat history", timestamp: new Date() - 1 }]}
      agent_replies={[{ msg: "Reply history", timestamp: new Date() }]}
    />
  );
};

export const Processing = () => {
  return <AgentThinking stateManager={testStore} />;
};

export const Labeling = () => {
  return (
    <Question
      stateManager={testStore}
      failidx={0}
      chats={[{ msg: "Dig a blue hole in the sky", timestamp: new Date() }]}
    />
  );
};
