/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import ReactDOM from "react-dom";

import GoldenLayout from "golden-layout";
import "golden-layout/src/css/goldenlayout-base.css";
import "golden-layout/src/css/goldenlayout-dark-theme.css";

import MainPane from "./MainPane";
import InteractApp from "./components/Interact/InteractApp";
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
import Retrainer from "./components/Retrainer";
import OfflinePanel from "./components/OfflinePanel";
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
          type: "column",
          content: [
            {
              type: "stack",
              content: [
                {
                  title: "Interact",
                  type: "react-component",
                  component: "InteractApp",
                  props: { stateManager: stateManager },
                },
              ],
              height: 40,
            },
            {
              title: "Live Viewer",
              type: "react-component",
              component: "MainPane",
              props: { stateManager: stateManager },
            },
          ],
          width: 40,
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
                      cssClass: "scrollable",
                      type: "react-component",
                      component: "DashboardTimeline",
                      props: { stateManager: stateManager },
                    },
                    {
                      type: "row",
                      content: [
                        {
                          title: "Results",
                          cssClass: "scrollable",
                          type: "react-component",
                          component: "TimelineResults",
                          props: { stateManager: stateManager },
                        },
                        {
                          type: "stack",
                          id: "timelineDetails",
                          // empty content to be populated with detail panes on click
                          content: [],
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
                {
                  title: "Retrainer",
                  type: "react-component",
                  component: "Retrainer",
                  props: { stateManager: stateManager },
                },
                {
                  title: "Offline",
                  type: "react-component",
                  component: "OfflinePanel",
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
dashboardLayout.registerComponent("InteractApp", InteractApp);
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
dashboardLayout.registerComponent("Retrainer", Retrainer);
dashboardLayout.registerComponent("OfflinePanel", OfflinePanel);

// allows for css styling, e.g. to be scrollable
dashboardLayout.on("itemCreated", function (item) {
  if (item.config.cssClass) item.element.addClass(item.config.cssClass);
});

dashboardLayout.init();
stateManager.dashboardLayout = dashboardLayout;
