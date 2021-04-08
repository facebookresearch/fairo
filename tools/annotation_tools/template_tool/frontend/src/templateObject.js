import React from 'react';
import ReactDOM from 'react-dom';
import LogicalForm from './logicalForm.js'
import TextCommand from './textCommand.js'
import ListComponent from './listComponent.js'

// Renders an Annotation component consisting of a region for displaying text and area for logical form annotation
class TemplateAnnotator extends React.Component {

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
      this.handleTextChange = this.handleTextChange.bind(this);
      this.logSerialized = this.logSerialized.bind(this);
      this.componentDidMount = this.componentDidMount.bind(this);
      this.callAPI = this.callAPI.bind(this);
      this.updateLabels = this.updateLabels.bind(this);
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

    handleTextChange(e) {
      this.setState({ textValue: e.target.value });
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
  
    updateLabels(e) {
      // Make a shallow copy of the items
      try {
        // First check that the string is JSON valid
        let JSONActionDict = JSON.parse(this.state.value)
        let items = { ...this.state.dataset };
        items[this.state.textValue] = JSONActionDict
        // Set state to the data items
        this.setState({ dataset: items }, function () {
          try {
            let actionDict = JSONActionDict
            let JSONString = {
              "command": this.state.textValue,
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
  
  
    render() {
      return (
        <div style={{ padding: 10 }}>
          <b> {this.props.title} </b>
          <ListComponent fullText={this.props.fullText} onChange={this.handleTextChange} />
          <LogicalForm title="Action Dictionary List" currIndex={this.state.fragmentsIndex} value={this.state.value} onChange={this.handleChange} updateCommand={this.updateCommand} schema={this.props.schema} dataset={this.state.dataset} />
          <div onClick={this.logSerialized}>
            <button>Save</button>
          </div>
        </div>
      )
    }
  }

  export default TemplateAnnotator;