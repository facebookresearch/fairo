/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import Memory2D from "./Memory2D";
import testStore from "../TestStore";

export default {
  component: Memory2D,
  title: "Memory2D",
  // Our exports that end in "Data" are not stories.
  excludeStories: /.*Data$/,
};

export const Default = () => {
  return <Memory2D height={320} width={320} stateManager={testStore} />;
};

export const Loading = () => {
  return <Memory2D height={320} width={320} />;
};
