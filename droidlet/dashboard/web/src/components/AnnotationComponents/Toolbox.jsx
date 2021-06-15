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

    this.addMaskHandler = this.addMaskHandler.bind(this);
  }

  componentDidMount() {}

  render() {
    return (
      <div>
        <button onClick={this.addMaskHandler}>Add mask</button>
        <button onClick={this.deleteMaskHandler}>Delete mask</button>
      </div>
    );
  }

  addMaskHandler() {
    console.log("adding mask");
    this.props.addMaskHandler();
  }

  deleteMaskHandler() {
    console.log("delete mask");
  }
}

export default Toolbox;
