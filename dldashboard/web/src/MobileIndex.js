/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import ReactDOM from "react-dom";

import GoldenLayout from "golden-layout";
import "golden-layout/src/css/goldenlayout-base.css";
import "golden-layout/src/css/goldenlayout-dark-theme.css";

import Memory2D from "./components/Memory2D";
import stateManager from "./StateManager";
import LiveImage from "./components/LiveImage";
import InteractApp from "./components/Interact/InteractApp";

import "./index.css";

export function mobileLayout() {
  window.React = React;
  window.ReactDOM = ReactDOM;
  var config = {
    settings: {
      showPopoutIcon: false,
    },
    content: [
      {
        type: "column",
        content: [
          {
            type: "row",
            content: [
              {
                title: "rgb",
                type: "react-component",
                component: "rgb",
                props: {
                  stateManager: stateManager,
                  type: "rgb",
                  height: 320,
                  width: 320,
                  offsetH: 0,
                  offsetW: 0,
                },
              },
              {
                title: "Memory 2D",
                type: "react-component",
                component: "Memory2D",
                props: { stateManager: stateManager },
              },
            ],
          },
          {
            title: "InteractApp",
            type: "react-component",
            component: "InteractApp",
            props: { stateManager: stateManager },
          },
        ],
      },
    ],
  };

  var dashboardLayout = new GoldenLayout(config);
  dashboardLayout.registerComponent("rgb", LiveImage);
  dashboardLayout.registerComponent("Memory2D", Memory2D);
  dashboardLayout.registerComponent("InteractApp", InteractApp);
  dashboardLayout.init();
  stateManager.dashboardLayout = dashboardLayout;
}
