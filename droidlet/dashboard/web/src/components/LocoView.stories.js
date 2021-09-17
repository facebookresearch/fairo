/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LocoView.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import LocoView from "./LocoView";
import testStore from "../TestStore";

export default {
  component: LocoView,
  title: "LocoView",
  // Our exports that end in "Data" are not stories.
  excludeStories: /.*Data$/,
};

export const Default = () => {
  return (
    <LocoView
      stateManager={testStore}
    />
  );
};
