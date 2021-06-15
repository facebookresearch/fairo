/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

/* Props

prop_name (Type)
    Description

*/

class Toolbox extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <button onClick={this.props.addMaskHandler}>Add mask</button>
        <button onClick={this.props.deleteMaskHandler}>Delete mask</button>
        <button onClick={this.props.deleteLabelHandler}>Delete label</button>
        <button onClick={this.props.changeTextHandler}>Modify labels</button>
      </div>
    );
  }
}

export default Toolbox;
