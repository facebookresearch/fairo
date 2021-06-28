/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import ObjectAnnotation from "./AnnotationComponents/ObjectAnnotation";
import MobileObjectAnnotation from "./AnnotationComponents/MobileObjectAnnotation";

class ObjectFixup extends React.Component {
  constructor(props) {
    super(props);
    if (this.props.image) {
      this.initialState = {
        image: this.props.image,
      };
    } else {
      this.initialState = {
        image: undefined,
      };
    }
    this.state = this.initialState;
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    if (this.state.image === undefined) {
      return null;
    }

    if (this.props.isMobile) {
      return (
        <MobileObjectAnnotation
          image={this.state.image}
          objects={this.state.objects}
          not_turk={true}
          stateManager={this.props.stateManager}
          isMobile={this.props.isMobile}
          imageWidth={this.props.imageWidth}
        />
      );
    }
    return (
      <ObjectAnnotation
        image={this.state.image}
        objects={this.state.objects}
        not_turk={true}
        stateManager={this.props.stateManager}
        isMobile={this.props.isMobile}
        imageWidth={this.props.imageWidth}
      />
    );
  }
}

export default ObjectFixup;
