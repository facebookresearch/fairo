/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LiveHumans.stories.js

import React from "react";
import { action } from "@storybook/addon-actions";

import LiveHumans from "./LiveHumans";
import testStore from "../TestStore";

export default {
  component: LiveHumans,
  title: "LiveHumans",
  // Our exports that end in "Data" are not stories.
  excludeStories: /.*Data$/,
};

export const Default = () => {
  return (
    <LiveHumans
      height={320}
      width={320}
      offsetH={10}
      offsetW={10}
      stateManager={testStore}
    />
  );
};

export const Loading = () => {
  return <LiveHumans height={320} width={320} offsetH={10} offsetW={10} />;
};
