# Memory2D for Droidlet

This component describes a visual mapping of objects in an agent's memory.
The map is disabled by default, use the `--draw_map` flag with the agent to enable.

## Relevant Options

- `--draw_map` enables a toggle that allows the map to be seen in the dashboard.
- `--map_update_ticks <ticks>` (CraftAssist only) changes the number of `<ticks>` after which the map will update itself (default 20 ticks = 1 second).

## Features

- Zoomable, draggable map that plots the agent, obstacles, and all objects with positional attributes from memory
- Can edit memory tags of and inspect triples associated with objects
- Ability to select objects, which users can then group together (i.e. these points describe a chair)
- Sidebar menu with options to toggle dynamic positioning, map view, map size, etc.

### Keybindings/Shortcuts

- `Cmd` (or `Ctrl` for Windows) - Enters "selection mode", where you can click on any object to select it
- `Esc` - Closes any tabular components or the Memory2D menu
- `Right Click` - Enters "drawing mode", where you can describe a rectangle that selects/unselects all objects within drawn frame.
- `Middle Click` - Opens/closes Memory2D menu. Also three finger tap on some trackpads.

## Demos

Note: these demos are recorded using a build from July 2022

Note: Memory2D may appear to have significantly more lag due to GIF formatting

### Initialization

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=120iYksCeyQz8hOC1rb3ua8ebNLRAUkOV" width="500" alt="Starting Memory2D"/> </br>
</p>

To get started using Memory2D on the dashboard, ensure you run your agent with the `--draw_map` option. If this option is triggered, a "Toggle Map" button should appear in the dashboard Settings pane. If you happen to start the dashboard before the agent has fully loaded (i.e before the agent starts perceiving its world), refresh the page and the "Toggle Map" button should appear.

### Interfacing and Inspection

Every clickable point on the map will open some sort of tabular component. Broadly, there are two types of points:

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1TuoQsfqq7A-jLiQYM-OLzLhkQHHBXt9M" width="300" alt="Two Types of Clickable Points"/> </br>
    <b>Cluster</b> and <b>Objects</b> (<i>Bot</i> and <i>Node</i>)
</p>

- **Objects** - These have no numeric label. Each represents a tuple in agent memory (i.e. a `ReferenceObject`), and clicking on an **Object** will open up a `MemoryMapTable` that allows a user to inspect and edit the memory tags associated with that **Object**. If desired, a user can enable "Show Triples" from the side menu to inspect those as well.
<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1lz40BIBK_Vn6YH_K63oJgY9se5vdUEMd" width="500" alt="Clicking Object"/> </br>
    Note that both the points clicked here are objects, the second being </br> a specially designed point that represents the agent itself.
</p>

- **Clusters** - These are signified with a number label in their center, representing the amount of objects in that cluster. Composed of all the objects that have nearby coordinates in the current map view. For example, suppose the map is in the XY view and there exist objects at (0, 0, 64), (0, 0, 63), and (0.1, -0.23, -743); these three objects would all appear in the same cluster. </br>
Clicking on a cluster will open up a `ClusteredObjsPopup`. Each row of this popup represents an **Object** which you can interact with as if were its own plot point (i.e. you can click
to open an associated `MemoryMapTable`, can "`Cmd` + click` to select, etc.)
<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1N1fkbK0kxWA80_IozJogptoOmoXXptxA" width="500" alt="Clicking Cluster"/>
</p>

### Editing Memory Tags

By simply clicking on an object, a `MemoryMapTable` will appear with several editable text fields.
Some fields are immutable (i.e. `memid`), while others are user-editable.
Clicking the "Submit" button at the bottom of the table will send the changes to the agent backend, which may take a few moments to register the manual changes.
Incorrectly formatted changes are automatically caught, and users can refresh all changed values to what they were when first opening the table. Moreover, users can restore an object to eliminate any manually made changes to it.

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1Y4WlYLd2MLGaar7vCZVzWfFSEeHkIiR3" width="500" alt="Editing Memory Tags"/>
</p>

The manual changes made by a user are _persistent_ and will continuously overwrite any other changes made to the agent memory.
This may cause some strange behaviors; for example, if a user changed the position of a `ModNode`, that object will appear static on the map even if it is moving around in the world.
It is helpful to note that Memory2D is a view of the world _as the agent sees it_ rather than what the world actually is.
Editing memory tags is akin to "brainwashing" the agent (i.e. "the cow is actually a chicken" or "my name is not bot.9059295.83").
Luckily, this can be undone with the restore feature.

### Selecting and Grouping

Users have two options to select objects:

- `Right Click` -> Drag -> `Right Click` (**Preferred**) </br>
Use `Right Click` (or equivalent on trackpad) to enter "drawing mode". You can then drag, zoom, and/or pan to describe a rectangle in which all objects in the drawn frame will be selected/unselected. `Right Click` again to finish drawing and toggle the select/unselect actions. Can also use `Esc` to exit "drawing mode" without toggling any actions in the frame you already drew.
<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1vt-MnLCnlTPVRoEhDOEsMbC8W5DLSjaH" width="500" alt="Group with Drag + Click"/>
</p>

- `Cmd` + Click </br>
Similar to selection in most file managers and visual media software, press `Cmd` (or `Ctrl` on Windows) and (left) click an object to select it. Use this option for finer selection of objects.
<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1SnQfhRzJ0fLdJGTOz_WnWTx0iwVHDq1e" width="500" alt="Group with Cmd + Click"/>
</p>

### Menu Tools

The `Memory2DMenu` provides several tools for the user.
Hover over the help icons for the most updated guidance.

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1VtLlG0mOo-iSbxmlV0rMISZrCkdbMczU" width="500" alt="Memory2DMenu"/>
</p>

Descriptions are listed in order of appearance from top:

- Grouping Form: used to group all selected objects by storing "is_a" triples in memory. Using an invalid string (i.e. anything that would fail `JSON.stringify`) will cause agent crash. Currently only functional for CraftAssist.
- Map Options: used to customize UX. For example, toggle `Square Map` if you plan on using `Center to Bot` often.
- View Options: to inspect objects in other dimensional views. Useful when several objects are clustered in one view (i.e. stacked on top of one another) and different view reveals underlying structure.
- Node Details: Short summary of all plotted objects and their node types. Users can change colors of different nodes to increase/reduce visibility of certain types of nodes.

### Quirks

- Tabular components will have wonky positioning when zooming in. If you are unable to see a tabular component, simply zoom all the way out for it come into view.
- Clicking the <span style="color:red">X</span> on any tabular component will cause you to unfocus from the Memory2D div. This is usually irrelevant, except when using `Cmd` + Click to select objects immediately after exiting by clicking <span style="color:red">X</span>. Best practice to avoid this behavior is to use `Esc` to close tabular components.

## Libraries

### react-konva

The backbone of the Memory2D map is `react-konva`, a canvas graphics drawing library with bindings to the Konva framework.

All of the actual map itself is contained in a Konva `Stage`. The `Stage` serves as the parent component for several `Layer`s (i.e. `coordinateAxesLayer`, `renderedObjects`, etc.). Each `Layer` contains a bevy of different Konva objects, all of which are highly and easily customizable.

Documentation: https://konvajs.org/api/Konva.html

### Material-UI

Memory2D uses several Material-UI (v4) components to provide a smooth user experience to interface with Memory2D.

The two most relevant use cases of this library are with the tabular components and the Memory2D menu. The tabular components (i.e. `MemoryMapTable`, `ClusteredObjsPopup`) describe data about a figure on the Konva `Stage`. These tabular components are rendered on top of the `Stage` and are positioned using raw Javascript and absolute coordinates.

Documentation: https://v4.mui.com/
