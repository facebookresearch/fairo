/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/MemoryList.js

import React from "react";

import ReactVirtualizedTable from "./Memory/MemoryVirtualizedTable";
import MemoryManager from "./Memory/MemoryManager";
import MemoryDetail from "./Memory/MemoryDetail";
import TextField from "@material-ui/core/TextField";
import Drawer from "@material-ui/core/Drawer";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Divider from "@material-ui/core/Divider";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

var hashCode = function (s) {
  return s.split("").reduce(function (a, b) {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);
};

class MemoryList extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      height: 400,
      width: 600,
      isLoaded: false,
      memory: null,
      filter: "",
      showDetail: false,
      detailUUID: null,
    };
    this.state = this.initialState;
    this.outer_div = React.createRef();
    this.resizeHandler = this.resizeHandler.bind(this);
  }

  resizeHandler() {
    if (this.outer_div != null && this.outer_div.current != null) {
      let { clientHeight, clientWidth } = this.outer_div.current;
      if (
        (clientHeight !== undefined && clientHeight !== this.state.height) ||
        (clientWidth !== undefined && clientWidth !== this.state.width)
      ) {
        this.setState({ height: clientHeight, width: clientWidth });
      }
    }
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    if (this.props.glContainer !== undefined) {
      // if this is inside a golden-layout container
      this.props.glContainer.on("resize", this.resizeHandler);
    }
  }
  componentDidUpdate(prevProps, prevState) {
    this.resizeHandler();
  }

  render() {
    if (!this.state.isLoaded) return <p>Loading</p>;
    let { height, width, memory, isLoaded } = this.state;

    if (height === 0 && width === 0) {
      // return early for performance
      return (
        <div ref={this.outer_div} style={{ height: "100%", width: "100%" }} />
      );
    }

    if (!isLoaded) {
      return (
        <div ref={this.outer_div} style={{ height: "100%", width: "100%" }}>
          Loading...
        </div>
      );
    }

    const darkTheme = createMuiTheme({
      palette: {
        type: "dark",
      },
    });

    const memoryManager = new MemoryManager(memory, this.state.filter);

    const showMemeoryDetail = (memoryUUID) => {
      this.setState({ detailUUID: memoryUUID, showDetail: true });
    };

    const paddedHeight = this.state.height - 24;
    const paddedWidth = this.state.width - 24;

    // final render
    return (
      <ThemeProvider theme={darkTheme}>
        <div
          ref={this.outer_div}
          style={{ padding: 12, height: paddedHeight, width: paddedWidth }}
        >
          <TextField
            style={{
              borderBlockColor: "white",
            }}
            color="primary"
            id="outlined-uncontrolled"
            label="Search"
            margin="dense"
            variant="outlined"
            onChange={(event) => {
              this.setState({ filter: event.target.value });
            }}
          />
          <ReactVirtualizedTable
            height={paddedHeight}
            width={paddedWidth}
            memoryManager={memoryManager}
            onShowMemeoryDetail={showMemeoryDetail}
          />
          <Drawer anchor="right" open={this.state.showDetail}>
            <div style={{ width: 450 }}>
              <IconButton onClick={() => this.setState({ showDetail: false })}>
                <CloseIcon />
              </IconButton>
              <Divider />
              <MemoryDetail
                memoryManager={memoryManager}
                uuid={this.state.detailUUID}
              />
            </div>
          </Drawer>
        </div>
      </ThemeProvider>
    );
  }
}

export default MemoryList;
