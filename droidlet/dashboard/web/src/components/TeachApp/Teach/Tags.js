/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Component to add/remove tags from a block.
 */

import React, { useRef, useState } from "react";
import Button from "@material-ui/core/Button";
import Chip from "@material-ui/core/Chip";
import TextField from "@material-ui/core/TextField";
import { makeStyles } from "@material-ui/core/styles";

const useKeyGenerator = () => {
  // use ref to avoid stale closures
  const indexRef = useRef(0);
  const getNextKey = () => {
    indexRef.current++;
    return indexRef.current;
  };
  return { getNextKey };
};

const Tags = ({ tagsList, setTagsList }) => {
  const useStyles = makeStyles((theme) => ({
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

  const classes = useStyles();

  const [currentTag, setCurrentTag] = useState("");
  const { getNextKey } = useKeyGenerator();

  const handleDelete = (tag) => () => {
    setTagsList((currentList) =>
      currentList.filter((el) => el.tag !== tag.tag)
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // if current tag is not empty and not in tags list already, add it
    if (currentTag && !tagsList.find((t) => t.tag === currentTag)) {
      setTagsList((currentList) =>
        currentList.concat([{ tag: currentTag, key: getNextKey() }])
      );
    }
    setCurrentTag("");
  };

  return (
    <div>
      {/* use a form here to enable pressing enter to add a tag */}
      <form onSubmit={handleSubmit}>
        <TextField
          size="small"
          variant="outlined"
          label="Tag"
          InputLabelProps={{
            className: classes.formLabel,
          }}
          FormHelperTextProps={{
            className: classes.formHelperText,
          }}
          InputProps={{
            className: classes.formInput,
          }}
          helperText="Keywords that will find your command in searches."
          value={currentTag}
          onChange={(e) => setCurrentTag(e.target.value)}
          style={{
            maxWidth: "320px",
            marginRight: "20px",
            marginBottom: "10px",
          }}
        />
        <Button type="submit" variant="contained" color="secondary">
          Add Tag
        </Button>
        <span
          style={{
            display: "flex",
            maxHeight: "40px",
            overflow: "scroll",
            flexWrap: "wrap",
          }}
        >
          {tagsList.map((tag) => (
            <Chip
              label={tag.tag}
              onDelete={handleDelete(tag)}
              key={tag.key}
              style={{ marginLeft: "10px", marginBottom: "10px" }}
            />
          ))}
        </span>
      </form>
    </div>
  );
};

export default Tags;
