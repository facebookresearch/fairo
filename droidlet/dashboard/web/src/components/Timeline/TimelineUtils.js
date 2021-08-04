/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import Fuse from "fuse.js";

export function capitalizeEvent(str) {
  // replaces underscores with spaces
  str = str.replace(/_/g, " ");
  // capitalizes the first letter of every word
  return str.replace(/\w\S*/g, function (txt) {
    return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
  });
}

export function jsonToArray(eventObj) {
  // turns JSON hook data into an array that can easily be turned into an HTML table
  let tableArr = [];
  for (let key in eventObj) {
    // stringify JSON object for logical form
    if (key === "logical_form") {
      tableArr.push({
        event: capitalizeEvent(key),
        description: JSON.stringify(eventObj[key]),
      });
    } else {
      tableArr.push({
        event: capitalizeEvent(key),
        description: eventObj[key],
      });
    }
  }
  return tableArr;
}

export function handleClick(stateManager, item) {
  const eventObj = JSON.parse(item);
  let tableArr = jsonToArray(eventObj);
  stateManager.memory.timelineDetails = tableArr;
  stateManager.updateTimelineResults();

  var config = {
    title: capitalizeEvent(eventObj["name"]),
    cssClass: "scrollable",
    type: "react-component",
    component: "TimelineDetails",
    props: { stateManager: stateManager },
  };
  stateManager.dashboardLayout.root
    .getItemsById("timelineDetails")[0]
    .addChild(config);
}

export function handleSearch(stateManager, pattern) {
  const matches = [];
  if (pattern) {
    stateManager.memory.timelineSearchPattern = pattern;
    const fuseOptions = {
      // set ignoreLocation to true or else it searches the first 60 characters by default
      ignoreLocation: true,
      useExtendedSearch: true,
    };

    const fuse = new Fuse(
      stateManager.memory.timelineEventHistory,
      fuseOptions
    );

    // prepending Fuse operator to search for results that include the pattern
    const result = fuse.search("'" + pattern);

    if (result.length) {
      result.forEach(({ item }) => {
        const eventObj = JSON.parse(item);
        if (
          stateManager.memory.timelineFilters.includes(
            capitalizeEvent(eventObj["name"])
          )
        ) {
          matches.push(eventObj);
        }
      });
    }
  }
  stateManager.memory.timelineSearchResults = matches;
  stateManager.updateTimelineResults();
}
