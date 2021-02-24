/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "beautiful-react-diagrams/styles.css";
import Diagram, { createSchema, useSchema } from "beautiful-react-diagrams";

class ShowAgentComponents extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      action_dict: {},
    };
    this.state = this.initialState;
    // this.handleSubmit = this.handleSubmit.bind(this);
    // this.querySemanticParser = this.querySemanticParser.bind(this);
    this.qRef = React.createRef();
  }

  // querySemanticParser(res) {
  //     this.setState({
  //         action_dict: res.action_dict,
  //     });
  // }

  // handleSubmit(event) {
  //     var command = document.getElementById("command_input").innerHTML;
  //     this.props.stateManager.socket.emit("queryParser", {
  //         chat: command,
  //     });
  //     event.preventDefault();
  // }

  // componentDidMount() {
  //     if (this.props.stateManager) {
  //         this.props.stateManager.connect(this);
  //         this.props.stateManager.socket.on(
  //             "renderActionDict",
  //             this.querySemanticParser
  //         );
  //     }
  // }

  render() {
    const initialSchema = createSchema({
      nodes: [
        { id: "node-1", content: "Node 1", coordinates: [250, 60] },
        { id: "node-2", content: "Node 2", coordinates: [250, 160] },
        { id: "node-3", content: "Node 3", coordinates: [250, 220] },
        { id: "node-4", content: "Node 4", coordinates: [400, 200] },
        { id: "node-5", content: "Node-5", coordinates: [300, 260] },
      ],
      links: [
        { input: "node-1", output: "node-2" },
        { input: "node-2", output: "node-3" },
        { input: "node-3", output: "node-4" },
        {
          input: "node-3",
          output: "node-5",
          label: "Query Semantic Parser",
          readonly: true,
        },
      ],
    });

    const UncontrolledDiagram = () => {
      // create diagrams schema
      const [schema, { onChange }] = useSchema(initialSchema);

      return (
        <div style={{ height: "22.5rem" }}>
          <Diagram schema={schema} onChange={onChange} />
        </div>
      );
    };
    return <UncontrolledDiagram />;
  }
}

export default ShowAgentComponents;
