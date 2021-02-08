/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/MemoryList.js

import React from "react";

import ReactVirtualizedTable from "./Memory/MemoryVirtualizedTable";
import MemoryManager from "./Memory/MemoryManager";

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

    const memoryManager = new MemoryManager(memory);

    const showMemeoryDetail = (memoryUUID) => {
      if (this.props.stateManager && this.props.stateManager.dashboardLayout) {
        var newItemConfig = {
          title: "Memory Detail",
          type: "react-component",
          component: "MemoryDetail",
          props: { memoryManager, uuid: memoryUUID },
        };

        const layout = this.props.stateManager.dashboardLayout;

        var detailViewStack = {
          type: "stack",
          content: [],
        };

        if (layout.root.contentItems[0].contentItems.length < 3) {
          layout.root.contentItems[0].addChild(detailViewStack);
        }
        // add detail view
        layout.root.contentItems[0].contentItems[2].addChild(newItemConfig);
      }
    };

    console.log(this.state.height);

    const paddedHeight = this.state.height - 24;
    const paddedWidth = this.state.width - 24;

    // final render
    return (
      <div
        ref={this.outer_div}
        style={{ padding: 12, height: paddedHeight, width: paddedWidth }}
      >
        <ReactVirtualizedTable
          height={paddedHeight}
          width={paddedWidth}
          memoryManager={memoryManager}
          onShowMemeoryDetail={showMemeoryDetail}
        />
      </div>
    );
  }
}

export default MemoryList;
