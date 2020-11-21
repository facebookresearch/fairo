/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Base/root component for the Teach page and application.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { fade, makeStyles } from "@material-ui/core/styles";
import Button from "@material-ui/core/Button";
import Grid from "@material-ui/core/Grid";
import BasicModal from "./Modal";
import InputBase from "@material-ui/core/InputBase";
import TextField from "@material-ui/core/TextField";
import Typography from "@material-ui/core/Typography";
import SearchIcon from "@material-ui/icons/Search";
import { Link } from "react-router-dom";

import { addCommasBetweenJsonObjects } from "./Blockly/blocks/utils/json";
import Blockly from "blockly";
import BlocklyEditor, { Toolbox } from "./Blockly";
import BlocklyJS from "blockly/javascript";
import "./Blockly/blocks";
import LabelBlockModal from "./LabelBlockModal";
import Tags from "./Tags";

const DEFAULT_BLOCKLY_XML =
  '<xml xmlns="https://developers.google.com/blockly/xml"></xml>';

const useStyles = makeStyles((theme) => ({
  gridRoot: {
    backgroundColor: "#333333",
    maxHeight: "calc(50vh - 64px)",
    minHeight: "calc(50vh - 64px)",
  },
  right: {
    padding: theme.spacing(3, 5, 1),
    display: "flex",
    justifyContent: "space-between",
    flexDirection: "column",
    overflow: "scroll",
  },
  left: {
    padding: theme.spacing(3, 5, 1),
    display: "flex",
    justifyContent: "space-between",
    flexDirection: "column",
    overflow: "scroll",
  },
  savedCommandsTitle: {
    margin: theme.spacing(3, 5, 1),
  },
  search: {
    position: "relative",
    borderRadius: theme.shape.borderRadius,
    backgroundColor: "#fff",
    "&:hover": {
      backgroundColor: fade("#ffffff", 0.75),
    },
    marginRight: theme.spacing(2),
    marginLeft: 0,
    width: "100%",
    [theme.breakpoints.up("sm")]: {
      marginLeft: theme.spacing(3),
      width: "auto",
    },
  },
  searchIcon: {
    padding: theme.spacing(0, 2),
    height: "100%",
    position: "absolute",
    pointerEvents: "none",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  inputRoot: {
    color: "inherit",
  },
  inputInput: {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)}px)`,
    transition: theme.transitions.create("width"),
    width: "100%",
    color: "#000000",
    textAlign: "left",
    [theme.breakpoints.up("md")]: {
      width: "20ch",
    },
  },
  searchResult: {
    "&:hover": {
      backgroundColor: "#D8D8D8",
    },
    padding: theme.spacing(1, 2, 1),
    margin: theme.spacing(0, 3),
    cursor: "pointer",
  },
  searchResultList: {
    maxHeight: "calc(50vh - 154px)",
    overflow: "scroll",
    color: "#ffffff",
  },
  formLabel: {
    color: "#ffffff",
  },
  formHelperText: {
    color: "#ffffff",
  },
  formInput: {
    color: "#ffffff",
  },
}));

const CommandInfo = ({
  classes,
  commandText,
  setCommandText,
  logCode,
  saveCommand,
  statusMessage,
  tagsList,
  setTagsList,
}) => {
  return (
    <Grid item xs className={classes.left}>
      <div>
        <Typography
          variant="h5"
          style={{ marginBottom: "10px", color: "#ffffff" }}
        >
          Command Information
        </Typography>
        <TextField
          id="standard-basic"
          label="Command Text"
          InputLabelProps={{
            className: classes.formLabel,
          }}
          FormHelperTextProps={{
            className: classes.formHelperText,
          }}
          InputProps={{
            className: classes.formInput,
          }}
          helperText="The chat message that will make the bot perform your command."
          value={commandText}
          onChange={(e) => setCommandText(e.target.value)}
          style={{ paddingBottom: "20px", maxWidth: "500px", color: "#ffffff" }}
        />
        <Tags tagsList={tagsList} setTagsList={setTagsList} />
      </div>
      <div
        style={{
          marginBottom: "20px",
          alignSelf: "flex-end",
          display: "flex",
          alignItems: "center",
        }}
      >
        <Typography
          variant="body1"
          style={{
            marginRight: "10px",
            display: "inline",
            fontStyle: "italic",
          }}
        >
          {statusMessage}
        </Typography>
        <Button
          size="small"
          onClick={saveCommand}
          // disable disable for now for demo
          // disabled={!commandText}
          variant="contained"
          color="secondary"
          style={{ marginRight: "10px" }}
        >
          Save Command
        </Button>
        <Button
          size="small"
          onClick={logCode}
          variant="contained"
          color="secondary"
          style={{ marginRight: "10px" }}
        >
          Generate Code
        </Button>
        <Button size="small" variant="contained" color="secondary">
          <Link
            to="/teach_welcome"
            style={{ color: "inherit", textDecoration: "none" }}
            target="_blank"
            rel="noopener noreferrer"
          >
            View Docs
          </Link>
        </Button>
      </div>
    </Grid>
  );
};

const Search = ({
  classes,
  query,
  setQuery,
  addBlockToWorkspace,
  commandsList,
}) => {
  return (
    <Grid item xs className={classes.right}>
      <Typography variant="h5" className={classes.savedCommandsTitle}>
        Saved Commands
      </Typography>
      <div className={classes.search}>
        <div className={classes.searchIcon}>
          <SearchIcon />
        </div>
        <InputBase
          placeholder="Search…"
          classes={{
            root: classes.inputRoot,
            input: classes.inputInput,
          }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          inputProps={{ "aria-label": "search" }}
        />
      </div>
      <div className={classes.searchResultList}>
        {commandsList.length === 0 ? (
          <div>There are no commands yet.</div>
        ) : (
          [...commandsList].reverse().map((cmd) => (
            <div
              className={classes.searchResult}
              key={cmd.cmd_id}
              onClick={() => addBlockToWorkspace(cmd)}
            >
              {cmd.chat_message || "Untitled command"}
            </div>
          ))
        )}
      </div>
    </Grid>
  );
};

const Teach = ({ username, stateManager, updatedCommandList }) => {
  const [commandText, setCommandText] = useState("");
  const [query, setQuery] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [commandsList, setCommandsList] = useState([]);
  // list of "tags" of form { tag: <string>, key: <int> }
  const [tagsList, setTagsList] = useState([]);
  const [modalBlock, setModalBlock] = useState(null); // null == not open
  const [modalError, setModalError] = useState(null); // null == no special message
  const clearStatusTimeout = useRef(); // represents timeout which will clear status message
  const workspace = useRef();
  const [showModal, setShowModal] = useState(false);
  const [actionDictContent, setActionDictContent] = useState({});
  const classes = useStyles();

  const fetchCommands = useCallback(() => {
    const postData = {
      query: query,
    };
    stateManager.socket.emit("fetchCommand", postData);
    setCommandsList(updatedCommandList);
  }, [query]);

  // fetch existing saved commands
  useEffect(() => {
    fetchCommands();
  }, [fetchCommands]);

  // mount modal opening function to window so that Blockly can access it
  useEffect(() => {
    window.openSetLabelModal_ = (block, err = null) => {
      setModalBlock(block);
      setModalError(err);
    };
    window.saveBlockToDatabase_ = saveBlockToDatabase;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generateActionDict = () => {
    let code = BlocklyJS.workspaceToCode(workspace.current);
    const len = code.length;

    // remove Blockly auto-generated semicolon
    if (code.charAt(len - 1) === ";") {
      code = code.slice(0, len - 1);
    } else if (code.charAt(len - 2) === ";") {
      code = code.slice(0, len - 2) + code.slice(len - 1);
    }
    const validForm = `{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "actions": {
    "list": [${addCommasBetweenJsonObjects(code)}]
  }
}`;
    return validForm;
  };

  const generateWorkspaceBlockAssembly = () => {
    const blocks = workspace.current.getAllBlocks();
    let block = blocks.find((b) => b.type === "controls_command");
    if (!block) {
      console.error("Couldn't find a top-level command block.");
      return DEFAULT_BLOCKLY_XML;
    }
    block.type = "controls_savedCommand"; // make this able to be plugged into action sequences
    const blockAsText = Blockly.Xml.domToText(Blockly.Xml.blockToDom(block));
    const fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
    block.type = "controls_command"; // change back for normal interactions
    return fullBlockXML;
  };

  const addBlockToWorkspace = (block) => {
    // generate the block
    const dom = Blockly.Xml.textToDom(block.block_assembly);

    // append the block to the workspace
    const newBlockIds = Blockly.Xml.appendDomToWorkspace(
      dom,
      workspace.current
    );

    // set the label on the imported block
    const importedBlock = workspace.current.getBlockById(newBlockIds[0]);

    if (!importedBlock) return; // no block saved in command

    importedBlock.label = block.chat_message;

    // toggle in case the block was saved collapsed
    // this is required due to our custom collapse function
    importedBlock.setCollapsed(false);
    importedBlock.setCollapsed(true);

    // should be able to delete any block you imported
    importedBlock.setDeletable(true);
  };

  const logCode = () => {
    setShowModal(true);
    setActionDictContent(generateActionDict());
  };

  const closeModal = () => {
    setShowModal(false);
  };

  // make save request to backend
  const saveBlockToDatabase = (blockXML, label, action_dict = null) => {
    const postData = {
      chat_message: label,
      block_assembly: blockXML,
      username,
      tags_list: tagsList.map((tagEntry) => tagEntry.tag),
    };

    if (action_dict) postData.action_dict = action_dict;

    // if the current message is going to be cleared, reset it
    if (clearStatusTimeout.current) {
      clearTimeout(clearStatusTimeout.current);
      clearStatusTimeout.current = null;
    }

    if (blockXML === DEFAULT_BLOCKLY_XML) {
      setStatusMessage("Command block not found.");
      clearStatusTimeout.current = setTimeout(() => setStatusMessage(""), 5000); // remove message after 5s
      return;
    }

    // post command information to database
    stateManager.socket.emit("saveCommand", postData);
    return;
  };

  const saveCommand = () => {
    const action_dict = generateActionDict();
    const block_assembly = generateWorkspaceBlockAssembly();
    saveBlockToDatabase(block_assembly, commandText, action_dict);
    fetchCommands();
    setTagsList([]);
    setCommandText("");
  };

  return (
    <>
      <Grid container alignItems="flex-start" className={classes.gridRoot}>
        <CommandInfo
          classes={classes}
          commandText={commandText}
          setCommandText={setCommandText}
          logCode={logCode}
          saveCommand={saveCommand}
          statusMessage={statusMessage}
          tagsList={tagsList}
          setTagsList={setTagsList}
        />
        <Search
          classes={classes}
          query={query}
          setQuery={setQuery}
          addBlockToWorkspace={addBlockToWorkspace}
          commandsList={commandsList}
        />
      </Grid>
      <BlocklyEditor
        ref={workspace}
        readOnly={false}
        trashcan={true}
        move={{
          scrollbars: true,
          drag: true,
          wheel: true,
        }}
        initialXml={`
<xml xmlns="http://www.w3.org/1999/xhtml">
      <block type="controls_command"></block>
</xml>
`}
      >
        <Toolbox />
      </BlocklyEditor>
      <BasicModal
        open={showModal}
        close={closeModal}
        text_to_render={actionDictContent}
      />
      <LabelBlockModal
        open={modalBlock !== null}
        closeModal={() => setModalBlock(null)}
        block={modalBlock}
        error={modalError}
        saveBlockToDatabase={saveBlockToDatabase}
      />
    </>
  );
};

export default Teach;
