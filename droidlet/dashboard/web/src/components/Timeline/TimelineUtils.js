/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

export function capitalizeEvent(str) {
  // replaces underscores with spaces
  str = str.replace(/_/g, " ");
  // capitalizes the first letter of every word
  return str.replace(/\w\S*/g, function (txt) {
    return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
  });
}

export function renderTable(tableArr) {
  // returns an HTML table given an array
  if (tableArr) {
    return tableArr.map((data) => {
      const { event, description } = data;
      return (
        <tr>
          <td>
            <strong>{event}</strong>
          </td>
          <td>{description}</td>
        </tr>
      );
    });
  }
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
  stateManager.updateTimeline();

  var config = {
    title: capitalizeEvent(eventObj["name"]),
    cssClass: "scrollable",
    type: "react-component",
    component: "TimelineDetails",
    props: { stateManager: stateManager },
  };
  stateManager.dashboardLayout.root.contentItems[0].contentItems[1].contentItems[0].contentItems[5].contentItems[1].contentItems[1].addChild(
    config
  );
}
