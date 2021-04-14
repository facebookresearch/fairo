import React from 'react';
import ReactDOM from 'react-dom';
import TemplateAnnotator from './templateObject.js'
import LogicalForm from './logicalForm.js'
import TextCommand from './textCommand.js'

var baseSchema = require('./spec/grammar_spec.schema.json');
var filtersSchema = require('./spec/filters.schema.json');
var otherDialogueSchema = require('./spec/other_dialogue.schema.json');
var actionDictSchema = require('./spec/action_dict_components.schema.json');

// Renders two autocomplete components, one for full commands and one for fragments
class AutocompleteAnnotator extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      fullText: [],
      fragmentsText: [],
      dataset: {},
    }
    this.updateFullText = this.updateFullText.bind(this)
  }

  componentDidMount() {
    fetch("http://localhost:9000/readAndSaveToFile/get_commands")
      .then(res => res.text())
      .then((text) => { this.setState({ fullText: text.split("\n").filter(r => r !== "") }) })

    fetch("http://localhost:9000/readAndSaveToFile/get_fragments")
      .then(res => res.text())
      .then((text) => { this.setState({ fragmentsText: text.split("\n").filter(r => r !== "") }) })

    // Combine JSON schemas to use in autocomplete pattern patching
    var combinedSchema = Object.assign({}, baseSchema.definitions, filtersSchema.definitions, actionDictSchema.definitions, otherDialogueSchema.definitions)
    this.setState({ schema: combinedSchema })
  }

  updateFullText(text, index) {
    let items = { ...this.state.fullText };
    items[index] = items[index].replace("<t1>", text);
    this.setState({ fullText: items }, function () {
      console.log("Updated list of commands")
    })
  }

  render() {
    return (
      <div>
        <div style={{ float: 'left', width: '45%', padding: 5}}>
          <ParseTreeAnnotator title="Command" fullText={this.state.fullText} updateFullText={this.updateFullText} schema={this.state.schema} />
        </div>
        <div style={{ float: 'left', width: '45%', padding: 5}}>
          <TemplateAnnotator title="Add New Command" fullText={this.state.fragmentsText} schema={this.state.schema} />
        </div>
      </div>
    )
  }
}

// Renders an Annotation component consisting of a region for displaying text and area for logical form annotation
class ParseTreeAnnotator extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      value: '',
      currIndex: -1,
      dataset: {},
      fragment: ""
    }
    /* Array of text commands that need labelling */
    this.handleChange = this.handleChange.bind(this);
    this.logSerialized = this.logSerialized.bind(this);
    this.uploadData = this.uploadData.bind(this);
    this.incrementIndex = this.incrementIndex.bind(this);
    this.decrementIndex = this.decrementIndex.bind(this);
    this.componentDidMount = this.componentDidMount.bind(this);
    this.callAPI = this.callAPI.bind(this);
    this.goToIndex = this.goToIndex.bind(this);
    this.updateLabels = this.updateLabels.bind(this);
    this.updateCommand = this.updateCommand.bind(this);
  }

  componentDidMount() {
    fetch("http://localhost:9000/readAndSaveToFile/get_labels_progress")
      .then(res => res.json())
      .then((data) => { this.setState({ dataset: data }) })
      .then(() => console.log(this.state.dataset))
  }

  callAPI(data) {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    };
    fetch("http://localhost:9000/readAndSaveToFile/append", requestOptions)
      .then(
        (result) => {
          console.log(result)
          this.setState({ value: "" })
          alert("saved!")
        },
        (error) => {
          console.log(error)
        }
      )
  }

  writeLabels(data) {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    };
    fetch("http://localhost:9000/readAndSaveToFile/writeLabels", requestOptions)
      .then(
        (result) => {
          console.log("success")
          console.log(result)
          this.setState({ value: "" })
          alert("saved!")
        },
        (error) => {
          console.log(error)
        }
      )
  }

  handleChange(e) {
    this.setState({ value: e.target.value });
  }

  updateCommand(text) {
    console.log(text)
    // possibly do more to update commands
    // this.setState({fragment: text})
  }

  incrementIndex() {
    console.log("Moving to the next command")
    if (this.state.currIndex + 1 >= this.props.fullText.length) {
      alert("Congrats! You have reached the end of annotations.")
    }
    this.setState({ currIndex: this.state.currIndex + 1, value: JSON.stringify(this.state.dataset[this.props.fullText[this.state.currIndex + 1]] ?? {}) });
  }

  decrementIndex() {
    console.log("Moving to the next command")
    console.log(this.state.currIndex)
    this.setState({ currIndex: this.state.currIndex - 1, value: JSON.stringify(this.state.dataset[this.props.fullText[this.state.currIndex - 1]] ?? {}) });
  }

  goToIndex(i) {
    console.log("Fetching index " + i)
    this.setState({ currIndex: Number(i), value: JSON.stringify(this.state.dataset[this.props.fullText[i]] ?? {}) });
    console.log(this.state.dataset)
  }

  updateLabels(e) {
    // Make a shallow copy of the items
    try {
      // First check that the string is JSON valid
      let JSONActionDict = JSON.parse(this.state.value)
      let items = { ...this.state.dataset };
      items[this.props.fullText[this.state.currIndex]] = JSONActionDict;
      // Set state to the data items
      this.setState({ dataset: items }, function () {
        try {
          let actionDict = JSONActionDict
          let JSONString = {
            "command": this.props.fullText[this.state.currIndex],
            "logical_form": actionDict
          }
          console.log("writing dataset")
          console.log(this.state.dataset)
          this.writeLabels(this.state.dataset)
        } catch (error) {
          console.error(error)
          console.log("Error parsing JSON")
          alert("Error: Could not save logical form. Check that JSON is formatted correctly.")
        }
      });
    } catch (error) {
      console.error(error)
      console.log("Error parsing JSON")
      alert("Error: Could not save logical form. Check that JSON is formatted correctly.")
    }
  }

  logSerialized() {
    console.log("saving serialized tree")
    // First save to local storage
    this.updateLabels()
  }

  uploadData() {
    console.log("Postprocessing annotations")
    // First postprocess
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    };
    fetch("http://localhost:9000/readAndSaveToFile/uploadDataToS3", requestOptions)
      .then(
        (result) => {
          if (result.status == 200) {
            this.setState({ value: "" })
            alert("saved!")
          } else {
            alert("Error: could not upload data to S3: " + result.statusText + "\n Check the format of your action dictionary labels.")
          }
        },
        (error) => {
          console.error(error)
        }
      )
  }


  render() {
    return (
      <div style={{ padding: 10 }}>
        <b> {this.props.title} </b>
        <TextCommand fullText={this.props.fullText} currIndex={this.state.currIndex} incrementIndex={this.incrementIndex} decrementIndex={this.decrementIndex} prevCommand={this.incrementIndex} goToIndex={this.goToIndex} />
        <LogicalForm title="Action Dictionary" currIndex={this.state.fragmentsIndex} value={this.state.value} onChange={this.handleChange} updateCommand={this.updateCommand} schema={this.props.schema} dataset={this.state.dataset} />
        <div style={{ padding: 10 }} onClick={this.logSerialized}>
          <button>Save Annotations</button>
        </div>
        <div style={{ padding: 10 }} onClick={this.uploadData}>
          <button>Process and Save Dataset</button>
        </div>
      </div>
    )
  }
}

export default AutocompleteAnnotator;