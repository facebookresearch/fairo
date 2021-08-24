/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

class OfflinePanel extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
        filepath: "",
    }

    this.state = this.initialState;
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  handleChange(event) {
    this.setState({ filepath: event.target.value });
  }

  handleSubmit(event) {
    this.props.stateManager.goOffline(this.state.filepath);
    event.preventDefault();
  }

  render() { 
    return (
      <div style={{ margin: "30px" }}>
        <form onSubmit={this.handleSubmit}>
            <label>
                Image / depth folder path: 
                <input
                type="text"
                value={this.state.filepath}
                onChange={this.handleChange}
                />
            </label>
            <input type="submit" value="Go offline" />
        </form>
      </div>
    );
  }
}

export default OfflinePanel;
