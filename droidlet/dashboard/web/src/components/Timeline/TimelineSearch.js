/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import Fuse from "fuse.js";
import { capitalizeEvent } from "./TimelineUtils";

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
