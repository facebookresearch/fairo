/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "./DataEntry.css";
import TagSelector from "./TagSelector";

/*
Props:

x (number): 
  x-coodinate for where to draw the overlay
y (number): 
  y-coodinate for where to draw the overlay
label: (string): 
  object label name
tags ([string]): 
  text tags to describe the object
includeSubmitButton (bool): 
  whether or not to include a submit button (only for new objects)
deleteCallback (func): 
  callback to delete the object
onSubmit (func): 
  callback to update label name and tags of object
*/
class DataEntry extends React.Component {
  constructor(props) {
    super(props);

    this.tags = this.props.tags || [];
    this.submit = this.submit.bind(this);

    this.nameRef = React.createRef();
  }

  render() {
    if (this.props.isMobile) {
      return (
        <div className="data-entry-root data-entry-block-mobile">
          <input placeholder="Object Name" ref={this.nameRef}></input>
          <TagSelector
            tags={this.labels}
            update={(tags) => (this.tags = tags)}
          />
          <button
            className="data-entry-submit"
            onClick={this.submit.bind(this)}
          >
            Submit
          </button>
        </div>
      );
    }
    return (
      <div
        className="data-entry-root"
        style={{
          top: this.props.y + "px",
          left: this.props.x + "px",
          position: "fixed",
        }}
      >
        <input
          placeholder="Object Name"
          ref={this.nameRef}
          defaultValue={this.props.label || ""}
        />
        <TagSelector tags={this.tags} update={(tags) => (this.tags = tags)} />
        {this.props.includeSubmitButton ? (
          <button className="data-entry-submit" onClick={this.submit}>
            Submit
          </button>
        ) : null}
        {this.props.deleteCallback ? (
          <button
            className="data-entry-delete"
            onClick={this.props.deleteCallback}
          >
            Delete object (⌫)
          </button>
        ) : null}
      </div>
    );
  }

  submit() {
    let name = this.nameRef.current.value.trim();
    // Validation
    if (name.length === 0) {
      alert("You must enter the object's name");
      return;
    } else if (this.tags.length < 2) {
      alert(
        "Please enter at least 2 descriptive tags for the object. Examples include: color, shape, size, orientation, etc."
      );
      return;
    }
    this.props.onSubmit({
      name: name.toLowerCase(),
      tags: this.tags,
    });
  }
}

export default DataEntry;
