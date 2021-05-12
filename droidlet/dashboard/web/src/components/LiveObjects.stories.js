/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LiveObjects.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import LiveObjects from "./LiveObjects";
import testStore from "../TestStore";

export default {
  component: LiveObjects,
  title: "LiveObjects",
  // Our exports that end in "Data" are not stories.
  excludeStories: /.*Data$/,
};

export const Default = () => {
  return (
    <LiveObjects
      height={320}
      width={320}
      offsetH={10}
      offsetW={10}
      stateManager={testStore}
    />
  );
};

export const Loading = () => {
  return <LiveObjects height={320} width={320} offsetH={10} offsetW={10} />;
};
