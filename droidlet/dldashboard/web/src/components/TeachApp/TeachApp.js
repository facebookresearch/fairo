/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * App.js handles displaying/switching between different views (settings, message, and TODO: fail)
 */
import React, { Component } from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import "./TeachApp.css";
import Teach from "./Teach/Teach";
import TeachLandingPage from "./Teach/LandingPage";
import muiTheme from "./theme";
import { MuiThemeProvider as ThemeProvider } from "@material-ui/core/styles";

class TeachApp extends Component {
  constructor(props) {
    super(props);
    this.initialState = {
      username: "dashboard",
      updatedCommandList: [],
    };
    this.state = this.initialState;
    this.qRef = React.createRef();
    this.updateCommandList = this.updateCommandList.bind(this);
  }

  updateCommandList(res) {
    this.setState({
      updatedCommandList: res.commandList,
    });
  }

  componentDidMount() {
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      this.props.stateManager.socket.on(
        "updateSearchList",
        this.updateCommandList
      );
    }
  }

  render() {
    return (
      <Router ref={this.qRef}>
        <ThemeProvider theme={muiTheme}>
          <div ref={this.qRef}>
            <Switch>
              <Route path="/teach_welcome">
                <TeachLandingPage />
              </Route>
              <Route path="/">
                <Teach
                  username={this.state.username}
                  stateManager={this.props.stateManager}
                  updatedCommandList={this.state.updatedCommandList}
                />
              </Route>
            </Switch>
          </div>
        </ThemeProvider>
      </Router>
    );
  }
}

export default TeachApp;
