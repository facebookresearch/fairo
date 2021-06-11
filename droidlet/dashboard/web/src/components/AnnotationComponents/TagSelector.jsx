/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "./TagSelector.css";

class TagSelector extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      selectedTags: [],
      suggestions: [],
      currentValue: "",
    };

    this.inputRef = React.createRef();
  }

  render() {
    return (
      <div className="tag-selector">
        <div className="tag-holder">
          {this.state.selectedTags.map((tag, i) => {
            return (
              <span className="tag-selected" key={i}>
                {tag}
                <button
                  onClick={() => {
                    this.removeTag(tag);
                  }}
                >
                  X
                </button>
              </span>
            );
          })}
        </div>
        <input
          type="text"
          placeholder="Tags (press Enter to add)"
          ref={this.inputRef}
          onChange={this.update.bind(this)}
          onKeyDown={this.keyDown.bind(this)}
        />
        <div className="tag-suggestions">
          {this.state.suggestions.map((tag, i) => {
            return (
              <span key={i}>
                <button
                  onClick={() => {
                    this.addTag(tag);
                  }}
                >
                  {i}
                </button>
              </span>
            );
          })}
        </div>
      </div>
    );
  }

  update(e) {
    let searchValue = this.inputRef.current.value;
    if (searchValue.length === 0) {
      this.setState({
        suggestions: [],
      });
      return;
    }
    this.setState({
      suggestions: this.props.tags.filter((i) => {
        if (
          i.toLowerCase().slice(0, searchValue.length) === searchValue &&
          this.state.selectedTags.indexOf(i) === -1
        ) {
          return true;
        } else {
          return false;
        }
      }),
    });
  }

  removeTag(tag) {
    this.setState(
      {
        selectedTags: this.state.selectedTags.filter((i) => {
          return i !== tag;
        }),
      },
      () => {
        this.props.update(this.state.selectedTags);
      }
    );
  }

  addTag(tag) {
    tag = tag.toLowerCase();
    this.setState(
      {
        selectedTags: this.state.selectedTags.concat(tag),
        suggestions: [],
      },
      () => {
        this.props.update(this.state.selectedTags);
      }
    );
    this.inputRef.current.value = "";
  }

  keyDown(e) {
    if (e.key === "Enter") {
      if (this.inputRef.current.value.trim() !== "") {
        this.addTag(this.inputRef.current.value);
      }
    }
    this.setState({
      currentValue: this.inputRef.current.value,
    });
  }
}

export default TagSelector;
