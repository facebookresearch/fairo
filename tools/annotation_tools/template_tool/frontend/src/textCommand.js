import React from 'react';

// Represents a Text Input node
class TextCommand extends React.Component {
  constructor(props) {
    super(props)
    this.fullText = props.fullText
    this.state = {
      value: {},
      currIndex: 0,
      indexValue: 0,
      fragment: ""
    }
    this.incrementIndex = props.incrementIndex
    this.handleChange = this.handleChange.bind(this)
  }

  handleChange(e) {
    this.setState({ indexValue: e.target.value });
  }

  render() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 5, marginTop: 5 }}>
        <div>
          <button style={{ marginBottom: 5, marginRight: 5, marginTop: 5 }} onClick={this.props.decrementIndex}>Prev</button>
          <button style={{ marginBottom: 5, marginRight: 5, marginTop: 5 }} onClick={this.props.incrementIndex}>Next</button>
        </div>
        <div style={{ marginBottom: 20, marginTop: 5 }}>
          <span>Index: <input onChange={this.handleChange} value={this.props.currIndex} type="number"></input></span>
          <button onClick={(param) => this.props.goToIndex(this.state.indexValue)}> Go! </button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 5, marginTop: 5 }}>
          <textarea rows="2" cols="10" value={this.props.fullText[this.props.currIndex]} fullWidth={false} />
        </div>
      </div>
    )
  }
}

export default TextCommand;