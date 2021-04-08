import React from 'react';
import ReactDOM from 'react-dom';
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
          <ParseTreeAnnotator title="Fragments" fullText={this.state.fragmentsText} schema={this.state.schema} />
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
    console.log("Uploading Data to S3")
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
        <LogicalForm currIndex={this.state.fragmentsIndex} value={this.state.value} onChange={this.handleChange} updateCommand={this.updateCommand} schema={this.props.schema} dataset={this.state.dataset} />
        <div onClick={this.logSerialized}>
          <button>Save</button>
        </div>
        <div onClick={this.uploadData}>
          <button>Upload to S3</button>
        </div>
      </div>
    )
  }
}

// Represents a Text Input node
class LogicalForm extends React.Component {
  constructor(props) {
    super(props)
    this.keyPress = this.keyPress.bind(this)
    this.state = {
      fragment: ""
    }
  }

  keyPress(e) {
    // Hit enter
    if (e.keyCode == 13) {
      let autocompletedResult = e.target.value
      // Apply replacements
      let definitions = Object.keys(this.props.schema)
      console.log(definitions)
      definitions.forEach(node => {
        let node_def = this.props.schema[node]
        let properties_subtree = {}
        if (Object.keys(node_def).includes("properties")) {
          let node_properties = Object.keys(node_def["properties"])
          node_properties.forEach(key => {
            properties_subtree[key] = ""
          })
        } else if (Object.keys(node_def).includes("oneOf") || Object.keys(node_def).includes("anyOf") || Object.keys(node_def).includes("allOf")) {
          var child_options;
          if (Object.keys(node_def).includes("oneOf")) {
            child_options = node_def["oneOf"]
          } else if (Object.keys(node_def).includes("anyOf")) {
            child_options = node_def["anyOf"]
          } else if (Object.keys(node_def).includes("allOf")) {
            child_options = node_def["allOf"]
          }
          let node_properties = child_options.map(
            child_def => {
              return ("properties" in child_def ? Object.keys(child_def.properties) : []);
            }
          ).reduce(
            (x, y) => x.concat(y)
          )
          node_properties.forEach(key => {
            properties_subtree[key] = ""
          })
        }
        console.log(node + ": " + JSON.stringify(properties_subtree))
        autocompletedResult = autocompletedResult.replace('"' + node + '"' + ":  ", '"' + node + '"' + ": " + JSON.stringify(properties_subtree))
      }
      )
      // Insert fragments
      let commands = Object.keys(this.props.dataset)
      console.log(commands)
      commands.forEach(text => {
        // this.props.updateCommand(text)
        autocompletedResult = autocompletedResult.replace(text, JSON.stringify(this.props.dataset[text]))
      })
      // Apply replacements        
      console.log(JSON.stringify(autocompletedResult))
      var obj = JSON.parse(autocompletedResult);
      var pretty = JSON.stringify(obj, undefined, 4);
      console.log(pretty)

      e.target.value = pretty
    }
  }

  render() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 10, marginTop: 10 }} >
        <b> Action Dictionary </b>
        <textarea rows="20" cols="100" value={this.props.value} onKeyDown={this.keyPress} onChange={(e) => this.props.onChange(e)} fullWidth={false} />
      </div>
    )
  }
}

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


export default AutocompleteAnnotator;