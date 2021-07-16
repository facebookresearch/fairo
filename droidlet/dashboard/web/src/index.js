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
import stateManager from "./StateManager";
import ObjectFixup from "./components/ObjectFixup";
import MemoryDetail from "./components/Memory/MemoryDetail";
import {
  DashboardTimeline,
  TimelineResults,
  TimelineDetails,
} from "./components/Timeline";
import { isMobile } from "react-device-detect";

import "./index.css";

window.React = React;
window.ReactDOM = ReactDOM;

if (isMobile) {
  let url = window.location;
  url += "mobile.html";
  window.location.href = url;
}

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
                  title: "Timeline",
                  type: "column",
                  content: [
                    {
                      title: "Timeline",
                      type: "react-component",
                      component: "DashboardTimeline",
                      props: { stateManager: stateManager },
                    },
                    {
                      type: "stack",
                      content: [
                        {
                          title: "Results",
                          type: "react-component",
                          component: "TimelineResults",
                          props: { stateManager: stateManager },
                        },
                        {
                          title: "Details",
                          type: "react-component",
                          component: "TimelineDetails",
                          props: { stateManager: stateManager },
                        },
                      ],
                    },
                  ],
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
dashboardLayout.registerComponent("ObjectFixup", ObjectFixup);
dashboardLayout.registerComponent("MemoryDetail", MemoryDetail);
dashboardLayout.registerComponent("DashboardTimeline", DashboardTimeline);
dashboardLayout.registerComponent("TimelineResults", TimelineResults);
dashboardLayout.registerComponent("TimelineDetails", TimelineDetails);
dashboardLayout.init();
stateManager.dashboardLayout = dashboardLayout;
