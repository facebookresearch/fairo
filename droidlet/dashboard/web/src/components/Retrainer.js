/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import "status-indicator/styles.css";
import Slider from "rc-slider";
import "rc-slider/assets/index.css";

let slider_style = { width: 600, margin: 50 };
const labelStyle = { minWidth: "60px", display: "inline-block" };

class Navigator extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
        learningRate: 0.005, 
        trainSplit: 0.7,
        maxIters: 100, 
        modelMetrics: null,
    }

    this.onAnnotationSave = this.onAnnotationSave.bind(this);
    this.onRetrain = this.onRetrain.bind(this);
    this.onModelSwitch = this.onModelSwitch.bind(this);
    this.onLearningRateChange = this.onLearningRateChange.bind(this);
    this.onTrainSplitChange = this.onTrainSplitChange.bind(this);
    this.onMaxItersChange = this.onMaxItersChange.bind(this);
    this.state = this.initialState;
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  onAnnotationSave() {
    if (this.props.stateManager) {
      this.props.stateManager.saveAnnotations()
    }
  }

  onRetrain() {
    if (this.props.stateManager) {
      console.log('retraining detector...')
      let { learningRate, trainSplit, maxIters } = this.state
      this.props.stateManager.socket.emit("retrain_detector", {
        learningRate, 
        trainSplit, 
        maxIters, 
      })
    }
  }

  onModelSwitch() {
    if (this.props.stateManager) {
      console.log("switching model...")
      this.props.stateManager.socket.emit("switch_detector")
    }
  }

  onLearningRateChange(value) {
    this.setState({ learningRate: value })
  }

  onTrainSplitChange(value) {
    this.setState({ trainSplit: value })
  }

  onMaxItersChange(value) {
    this.setState({ maxIters: value })
  }

  render() {
    let updatedModelDiv = null;
    if (this.state.modelMetrics) {
      let segm = this.state.modelMetrics.segm
      let evalText = Object.keys(segm).map(key => <div>{key + ": " + segm[key]}</div>)
      updatedModelDiv = (
        <div>
          <div>New model trained!</div>
          Evalution: {evalText}
          <button onClick={this.onModelSwitch}>Switch</button>
        </div>
      )
    }

    return (
      <div>
        <div style={slider_style}>
            <label style={labelStyle}>Learning Rate: &nbsp;</label>
            <span>{this.state.learningRate}</span>
            <br />
            <br />
            <Slider
                value={this.state.learningRate}
                min={0}
                max={1}
                step={0.005}
                onChange={this.onLearningRateChange}
            />
        </div>
        <div style={slider_style}>
            <label style={labelStyle}>Train split: &nbsp;</label>
            <span>{this.state.trainSplit}</span>
            <br />
            <br />
            <Slider
                value={this.state.trainSplit}
                min={0}
                max={1}
                step={0.05}
                onChange={this.onTrainSplitChange}
            />
        </div>
        <div style={slider_style}>
            <label style={labelStyle}>Max iterations: &nbsp;</label>
            <span>{this.state.maxIters}</span>
            <br />
            <br />
            <Slider
                value={this.state.maxIters}
                min={50}
                max={1000}
                step={50}
                onChange={this.onMaxItersChange}
            />
        </div>
        <button onClick={this.onAnnotationSave}>Save Annotations</button>
        <button onClick={this.onRetrain}>Retrain</button>
        {updatedModelDiv}
      </div>
    );
  }
}

export default Navigator;
