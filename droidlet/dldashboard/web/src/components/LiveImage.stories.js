/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LiveImage.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import LiveImage from "./LiveImage";
import testStore from "../TestStore";

export default {
  component: LiveImage,
  title: "LiveImage",
  // Our exports that end in "Data" are not stories.
  excludeStories: /.*Data$/,
};

export const Default = () => {
  return (
    <LiveImage
      height={320}
      width={320}
      offsetH={10}
      offsetW={10}
      stateManager={testStore}
    />
  );
};

export const Depth = () => {
  return (
    <LiveImage
      type={"depth"}
      height={320}
      width={320}
      offsetH={10}
      offsetW={10}
      stateManager={testStore}
    />
  );
};

export const Loading = () => {
  return (
    <LiveImage
      type={"depth"}
      height={320}
      width={320}
      offsetH={10}
      offsetW={10}
    />
  );
};
