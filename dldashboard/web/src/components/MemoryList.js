/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/MemoryList.js

import React from "react";

import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import ReactVirtualizedTable from "./Memory/MemoryVirtualizedTable";
import MemoryManager from "./Memory/MemoryManager";

var hashCode = function (s) {
  return s.split("").reduce(function (a, b) {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);
};

const DEFAULT_SPACING = 12;

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
    let { memories, named_abstractions, reference_objects, triples } = memory;

    // console.log("Memories: ", memories.slice(0, 5));
    // console.log("named_abstractions: ", named_abstractions.slice(0, 5));
    // console.log("reference_objects: ", reference_objects.slice(0, 5));
    // console.log("triples: ", triples.slice(0, 5));

    // Triples Schema
    // uuid, subj, subj_text, pred, pred_text, obj, obj_text, confidence

    // Named Abstraction Columns
    // uuid, name

    // SELECT uuid, node_type, create_time, updated_time, attended_time, is_snapshot
    // FROM Memories;

    // SELECT uuid, eid, x, y, z, yaw, pitch, name, type_name, ref_type
    // FROM ReferenceObjects;

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

    // final render
    return (
      <div ref={this.outer_div} style={{ margin: 12 }}>
        <ReactVirtualizedTable memoryManager={memoryManager} />
      </div>
    );
  }
}

export default MemoryList;
