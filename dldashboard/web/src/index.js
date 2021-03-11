/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import ReactDOM from "react-dom";

import GoldenLayout from "golden-layout";
import "golden-layout/src/css/goldenlayout-base.css";
import "golden-layout/src/css/goldenlayout-dark-theme.css";

import MainPane from "./MainPane";
import Settings from "./components/Settings";
import History from "./components/History";
import VoxelWorld from "./components/VoxelWorld/VoxelWorld";
import TurkInfo from "./components/Turk/TurkInfo";
import stateManager from "./StateManager";

import "./index.css";

window.React = React;
window.ReactDOM = ReactDOM;

var config = {
  settings: {
    showPopoutIcon: false,
  },
  content: [
    {
      type: "row",
      content: [
        {
          title: "Live Viewer",
          type: "react-component",
          component: "MainPane",
          props: { stateManager: stateManager },
        },
        {
          type: "column",
          content: [
            {
              type: "stack",
              content: [
                {
                  title: "Chat History",
                  type: "react-component",
                  component: "History",
                  props: { stateManager: stateManager },
                },
              ],
            },
            {
              type: "stack",
              content: [
                {
                  title: "Info",
                  type: "react-component",
                  component: "TurkInfo",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Settings",
                  type: "react-component",
                  component: "Settings",
                  props: { stateManager: stateManager },
                },
              ],
            },
          ],
        },
      ],
    },
  ],
};

var dashboardLayout = new GoldenLayout(config);

dashboardLayout.registerComponent("MainPane", MainPane);
dashboardLayout.registerComponent("Settings", Settings);
dashboardLayout.registerComponent("TurkInfo", TurkInfo);
dashboardLayout.registerComponent("History", History);
dashboardLayout.registerComponent("VoxelWorld", VoxelWorld);

dashboardLayout.init();

stateManager.dashboardLayout = dashboardLayout;
