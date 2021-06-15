/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

/* Props

prop_name (Type)
    Description

*/

class Toolbox extends React.Component {
  render() {
    return (
      <div>
        <button onClick={this.props.addMaskHandler}>Add mask</button>
        <button onClick={this.props.deleteMaskHandler}>Delete mask</button>
        <button onClick={this.props.deleteLabelHandler}>Delete label</button>
        <button onClick={this.props.changeTextHandler}>Modify labels</button>
        <button onClick={this.props.insertPointHandler}>Insert point</button>
        <button onClick={this.props.deletePointHandler}>Delete point</button>
      </div>
    );
  }
}

export default Toolbox;
