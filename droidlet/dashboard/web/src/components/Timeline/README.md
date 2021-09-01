# Timeline for Droidlet

This component consists of an agent activity visualizer for the Droidlet dashboard app. Enable this when running the
agent using the following flags:

`--enable_timeline` allows you to view incoming events on the timeline component in the dashboard.

`--log_timeline` saves the timeline history data to a .txt file which can further be used to recreate the timeline.

## Features

- A scrollable, zoomable timeline visualizer that can display perception, dialogue, and interpreter events
- Supports search functionality and shows detailed activity panes for each event
- Allows users to filter events and search results

## Libraries and Frameworks

### vis.js

The timeline relies on the vis.js Timeline and DataSet frameworks for the rendering of the main visualizer component.

vis.js [groups](https://visjs.github.io/vis-timeline/docs/timeline/#groups) allows event types (e.g. Perception, Dialogue, and Interpreter) to be grouped together in a separate row on the timeline.

Other available [options](https://visjs.github.io/vis-timeline/docs/timeline/#Configuration_Options) allow for features such as tooltips, maximum window size, and automatic scrolling. Tooltip `template` allows for custom HTML formatting of tooltips. `zoomMax` sets the maximum zoom interval for the visible range in milliseconds (currently set to 24 hours). `rollingMode` centers the timeline until the user drags the timeline, causing a toggle button to appear on the right side of the timeline.

Documentation: https://visjs.github.io/vis-timeline/docs/timeline/

### Fuse.js

This component also uses the Fuse.js fuzzy searching library to support searching through timeline events.

Fuse.js contains [many options](https://fusejs.io/api/options.html) for optimizing search, such as weighting searches by location or sorting results by score. `useExtendedSearch: true` enables the use of unix-like search commands. `ignoreLocation: true` searches the entire string for the specified pattern.

Documentation: https://fusejs.io/
