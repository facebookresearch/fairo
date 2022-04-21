/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LocoView.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import InteractApp from "./InteractApp";
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
