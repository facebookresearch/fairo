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
import InteractApp from "./components/Interact/InteractApp";

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
          type: "column",
          content: [
            {
              title: "Interact",
              type: "react-component",
              component: "InteractApp",
              props: { stateManager: stateManager },
            },
          ],
          width: 35,
        },
        {
          type: "column",
          content: [
            {
              type: "stack",
              content: [
                {
                  title: "VoxelWorld",
                  type: "react-component",
                  component: "VoxelWorld",
                  props: { stateManager: stateManager },
                },
              ],
              height: 64,
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
dashboardLayout.registerComponent("InteractApp", InteractApp);

dashboardLayout.init();

stateManager.dashboardLayout = dashboardLayout;
