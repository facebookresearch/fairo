/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import ReactDOM from "react-dom";

import GoldenLayout from "golden-layout";
import "golden-layout/src/css/goldenlayout-base.css";
import "golden-layout/src/css/goldenlayout-dark-theme.css";

import MainPane from "./MainPane";
import Console from "./components/Console";
import Settings from "./components/Settings";
import Navigator from "./components/Navigator";
import Memory2D from "./components/Memory2D";
import MemoryList from "./components/MemoryList";
import QuerySemanticParser from "./components/QuerySemanticParser";
import History from "./components/History";
import TeachApp from "./components/TeachApp/TeachApp";
import VoxelWorld from "./components/VoxelWorld/VoxelWorld";
import stateManager from "./StateManager";
import ObjectFixup from "./components/ObjectFixup";
import MemoryDetail from "./components/Memory/MemoryDetail";

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
                  title: "Voxel World",
                  type: "react-component",
                  component: "VoxelWorld",
                  props: {
                    stateManager: stateManager,
                  },
                },
                {
                  title: "Memory 2D",
                  type: "react-component",
                  component: "Memory2D",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Memory List",
                  type: "react-component",
                  component: "MemoryList",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Console",
                  type: "react-component",
                  component: "Console",
                },
                {
                  title: "Chat History",
                  type: "react-component",
                  component: "History",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Query the Semantic Parser",
                  type: "react-component",
                  component: "QuerySemanticParser",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Program the assistant",
                  type: "react-component",
                  component: "TeachApp",
                  props: {
                    stateManager: stateManager,
                  },
                },
              ],
            },
            {
              type: "stack",
              content: [
                {
                  title: "Settings",
                  type: "react-component",
                  component: "Settings",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Navigator",
                  type: "react-component",
                  component: "Navigator",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Object Annotation Fixer",
                  type: "react-component",
                  component: "ObjectFixup",
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
dashboardLayout.registerComponent("Console", Console);
dashboardLayout.registerComponent("Settings", Settings);
dashboardLayout.registerComponent("Navigator", Navigator);
dashboardLayout.registerComponent("Memory2D", Memory2D);
dashboardLayout.registerComponent("MemoryList", MemoryList);
dashboardLayout.registerComponent("QuerySemanticParser", QuerySemanticParser);
dashboardLayout.registerComponent("History", History);
dashboardLayout.registerComponent("TeachApp", TeachApp);
dashboardLayout.registerComponent("VoxelWorld", VoxelWorld);
dashboardLayout.registerComponent("ObjectFixup", ObjectFixup);
dashboardLayout.registerComponent("MemoryDetail", MemoryDetail);

dashboardLayout.init();

stateManager.dashboardLayout = dashboardLayout;
