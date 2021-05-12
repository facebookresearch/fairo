/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Converted USAGE_TEACH.md document with some modifications to make it work in HTML.
 */

import React from "react";
import Button from "@material-ui/core/Button";
import { Link } from "react-router-dom";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  wrapper: {
    margin: "24px 72px 8px",
    maxWidth: "80vw",
    "&>:not(img)": {
      maxWidth: "1000px",
    },
  },
}));

const LandingPage = () => {
  const classes = useStyles();
  return (
    <div className={classes.wrapper}>
      <h1 id="teach-the-bot">Teach the Bot</h1>
      <br />
      <p>
        Welcome to the Teach tool{" "}
        <span role="img" aria-label="Smiling Emoji">
          😃{" "}
        </span>
        <br />
        <br />
        This tool is designed to provide users with a way to directly program
        the bot (or agent), allowing for another type of human-in-the-loop
        learning. The tool has a specific focus on conditional statements and
        booleans, using constructs like while...until loops to build out
        functionality for conditional control flow.
      </p>
      <p>
        <strong>The Basics</strong>: To use the tool, drag and drop pieces from
        the categories in the toolbox to program the bot. Then, save your
        command and upload it to the bot (upload feature coming soon).
      </p>
      <br />
      <h2 id="accessing-the-tool">Accessing the Tool</h2>
      <br />
      <Button variant="contained" color="secondary">
        <Link to="/teach" style={{ color: "inherit", textDecoration: "none" }}>
          Go to App
        </Link>
      </Button>
      <br />
      <br />
      <h2 id="parts-of-the-tool">Parts of the Tool</h2>
      <p>
        <img
          src="/files/teach_sections_labeled.png"
          alt="Labeled Sections of App"
          style={{ maxWidth: "100%" }}
        ></img>
      </p>
      <h3 id="1-command-info-panel">1. Command Info Panel</h3>
      <p>
        The command info panel, located in the top left, has a few important
        pieces:
      </p>
      <ol>
        <li>
          Command text: the chat message associated with the command being built
        </li>
        <li>
          Add tags: add searchable tags to your current command. Once your
          command is saved, you can type these tags in the search box to bring
          your command up in the results.
        </li>
        <li>
          Save command: saves the command currently being built in the workspace
          under the "command" block
        </li>
        <li>
          Generate code: generates and logs the logical form for the current
          command
        </li>
      </ol>
      <h3 id="2-command-search-panel">2. Command Search Panel</h3>
      <p>
        The command search panel, located in the top right, is where you can
        find the commands that you and other users have saved during their use
        of the tool. Type the name of the command, label, or tag in the search
        bar, and you&#39;ll see the command (and others with similar search
        terms) pop up in the results. You can then scroll through and select the
        block you want, and click it to repopulate it into the command builder.
      </p>
      <p>
        <em>New Commands</em>: If you want to build a fairly big new command,
        this is the place to start. Search for the functionality you need; it
        may already exist within the tool!
      </p>
      <h3 id="3-command-builder">3. Command Builder</h3>
      <p>
        The command builder, located in the bottom half of the screen, is the
        core of the tool. It uses a drag-and-drop interface to build commands.
      </p>
      <h4 id="toolbox-categories">Toolbox &amp; Categories</h4>
      <p>
        On the left of the command builder, the grey area represents the
        toolbox. This is the &quot;drag&quot; area of the tool, where you can
        find the pieces of commands and drag them into the workspace. The
        different blocks are organized into 6 different categories:
      </p>
      <ol>
        <li>Actions: actions the bot can take</li>
        <li>Booleans: conditions for control flow</li>
        <li>
          Control: blocks that affect what happens when, e.g. loops and if
          statements
        </li>
        <li>Location: blocks related to location in the bot&#39;s world</li>
        <li>
          Values: blocks that represent values, like 0 and &quot;hello&quot;
          <ul>
            <li>
              Also includes &quot;object&quot;-related blocks to represent
              objects in the bot&#39;s world, like mobs and block objects
            </li>
            <li>
              Block objects have their own subcategory due to them needing a
              wide variety of filters to work correctly
            </li>
          </ul>
        </li>
      </ol>
      <h4 id="workspace">Workspace</h4>
      <p>
        The workspace is the right-hand side of the command builder. It&#39;s
        the blank white space into which you can drag any of the blocks from the
        toolbox. To build a command, simply drag any block from the toolbox into
        this white space. Keep doing this and you&#39;ll have a command built!
      </p>
      <p>
        The workspace provides some other special features to enhance the
        drag-and-drop experience. They&#39;re listed below:
      </p>
      <p>
        <em>Command Block:</em> This block is part of every command you create.
        When you click "Save Command", the blocks inside here are the command
        steps which will actually get saved. Any blocks outside this will be
        ignored. You can't delete this block because it's necessary for saving
        to work.
      </p>
      <p>
        <em>Right-Click Menu:</em> Blocks provide a custom menu on right-click
        that provides some special features. Some of the features are
        self-explanatory clipboard options (duplicate, delete) and the others
        are outlined below.
      </p>
      <p>
        <em>Types:</em> Certain blocks only fit into certain spaces, although
        they may look like they fit anywhere. This is because blocks have{" "}
        <em>types</em>; a block that outputs a &quot;Location&quot; type
        won&#39;t fit into a block that accepts a &quot;Number&quot; type.
      </p>
      <p>
        <em>Labels and Collapsing Blocks:</em> Every block can have a label
        associated with it. Once you give a block a label, you can
        &quot;collapse&quot; it, changing its appearance from the block-by-block
        command you built to a compact block with a single text label on it. To
        do this, right-click on the block, then click &quot;Add Label&quot;, or
        &quot;Collapse Block&quot; if it already has a label. Don&#39;t worry,
        you can revert to the original state the same way, by right clicking the
        block and selecting &quot;Collapse Block&quot;. You can also save blocks
        you&#39;ve labeled directly to the database in the pop-up window
        generated by the &quot;Add Label&quot; option.
      </p>
      <p>
        <em>Clipboard Tools:</em> Each block can be &quot;highlighted&quot; by a
        click, allowing basic clipboard controls to be used on it. This includes
        control-C (command-C on Mac) for copy, control-V/command-V for paste,
        backspace/delete for delete, and control-X/command-X for cut.
      </p>
      <h3 id="4-nav-bar">4. Nav Bar</h3>
      <p>
        The nav bar is a simple tool to help one get around the site. Use the
        menu in the top left to access the Chat app, and the settings button in
        the top right to visit the settings page for the application.
      </p>
    </div>
  );
};

export default LandingPage;
