/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "./DataEntry.css";
import TagSelector from "./TagSelector";

/*
Props:
draw_location: 
    where to draw the overlay
    {x: x location, y: y location}

objectId:
    Object indentifier (int)
*/
class DataEntry extends React.Component {
  constructor(props) {
    super(props);

    this.labels = [];
    this.tags = [];

    this.nameRef = React.createRef();
  }

  render() {
    return (
      <div
        className="data-entry-root"
        style={{
          top: this.props.y + "px",
          left: this.props.x + "px",
          position: "fixed",
        }}
      >
        <input placeholder="Object Name" ref={this.nameRef}></input>
        <TagSelector tags={this.labels} update={(tags) => (this.tags = tags)} />
        <button className="data-entry-submit" onClick={this.submit.bind(this)}>
          Submit
        </button>
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
