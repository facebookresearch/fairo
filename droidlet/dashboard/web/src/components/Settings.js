/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import "status-indicator/styles.css";
import Slider from "rc-slider";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Switch from "@material-ui/core/Switch";
import "rc-slider/assets/index.css";
import stateManager from "../StateManager";

let slider_style = { width: 600, margin: 50 };
const labelStyle = { minWidth: "60px", display: "inline-block" };

class Settings extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      url: this.props.stateManager.url,
      fps: 0,
      connected: false,
      agent_enable_map: false,
      image_quality: -1,
      image_resolution: -1,
      use_old_annotation: stateManager.useDesktopComponentOnMobile,
    };

    if (this.props.isMobile) {
      slider_style = { margin: 50 };
    }

    this.state = this.initialState;
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.setImageSettings = this.setImageSettings.bind(this);
    this.onImageQualityChange = this.onImageQualityChange.bind(this);
    this.onImageResolutionChange = this.onImageResolutionChange.bind(this);
  }

  setImageSettings(newSettings) {
    this.setState(newSettings);
  }

  handleChange(event) {
    this.setState({ url: event.target.value });
  }

  handleSubmit(event) {
    this.props.stateManager.setUrl(this.state.url);
    event.preventDefault();
  }

  handleClearSettings(event) {
    this.props.stateManager.setDefaultUrl();
  }

  componentDidMount() {
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      this.setState({ connected: this.props.stateManager.connected });
      this.props.stateManager.socket.on(
        "image_settings",
        this.setImageSettings
      );
    }
  }

  componentWillUnmount() {
    if (this.props.stateManager) {
      this.props.stateManager.disconnect(this);
      this.setState({ connected: false });
      this.props.stateManager.socket.off(
        "image_settings",
        this.setImageSettings
      );
    }
  }

  onImageQualityChange(value) {
    if (this.state.connected === true) {
      this.props.stateManager.socket.emit("update_image_settings", {
        image_quality: value,
        image_resolution: this.state.image_resolution,
      });
    }
  }

  onImageResolutionChange(value) {
    if (this.state.connected === true) {
      this.props.stateManager.socket.emit("update_image_settings", {
        image_quality: this.state.image_quality,
        image_resolution: value,
      });
    }
  }

  // switch which annotation component to use
  switchAnnotation = (event) => {
    stateManager.useDesktopComponentOnMobile = event.target.checked;
    this.setState({
      use_old_annotation: stateManager.useDesktopComponentOnMobile,
    });
  };

  render() {
    var status_indicator;
    if (this.state.connected === true) {
      status_indicator = <status-indicator positive pulse></status-indicator>;
    } else {
      status_indicator = <status-indicator negative pulse></status-indicator>;
    }

    var map_switch = <p />;
    if (this.state.agent_enable_map === true) {
      map_switch = (
        <FormControlLabel
          onChange={(e) => {
            this.props.stateManager.handleMapToggle();
          }}
          control={<Switch size="small" />}
          label="Toggle Dashboard: "
          labelPlacement="start"
        />
      );
    }

    var image_slider_disabled = false;
    if (this.state.image_quality === -1) {
      image_slider_disabled = true;
    }

    return (
      <div>
        <form onSubmit={this.handleSubmit}>
          <label>
            Server / Robot URL:
            <input
              type="text"
              value={this.state.url}
              onChange={this.handleChange}
            />
          </label>
          <input type="submit" value="Reconnect" />
        </form>
        <form onSubmit={this.handleClearSettings}>
          <input type="submit" value="Clear Saved Settings" />
        </form>

        {this.props.isMobile && (
          <FormControlLabel
            control={
              <Switch
                checked={stateManager.useDesktopComponentOnMobile}
                onChange={this.switchAnnotation}
              />
            }
            label="Use Old Annotation Tool"
          />
        )}

        <p> FPS: {this.state.fps} </p>
        <p> Connection Status: {status_indicator} </p>
        <p>
          <div style={{ position: "absolute", left: "-15px" }}>
            {map_switch}
          </div>
        </p>
        <div style={slider_style}>
          <label style={labelStyle}>Image Quality: &nbsp;</label>
          <span>{this.state.image_quality}</span>
          <br />
          <br />
          <Slider
            value={this.state.image_quality}
            min={1}
            max={100}
            step={1}
            onChange={this.onImageQualityChange}
            disabled={image_slider_disabled}
          />
        </div>
        <div style={slider_style}>
          <label style={labelStyle}>Image Resolution: &nbsp;</label>
          <span>{this.state.image_resolution}</span>
          <br />
          <br />
          <Slider
            value={this.state.image_resolution}
            min={16}
            max={512}
            step={16}
            onChange={this.onImageResolutionChange}
            disabled={image_slider_disabled}
          />
        </div>
      </div>
    );
  }
}

export default Settings;
