import React from 'react';
import ReactDOM from 'react-dom';
import LogicalForm from './logicalForm.js'
import TextCommand from './textCommand.js'
import ListComponent from './listComponent.js'
import { Autocomplete } from '@material-ui/lab';
import { TextField } from '@material-ui/core';

// Renders an Annotation component consisting of a region for displaying text and area for logical form annotation
class TemplateAnnotator extends React.Component {

    constructor(props) {
      super(props);
      this.state = {
        command: "",
        value: "",
        name: "",
        currIndex: -1,
        dataset: {},
        templates: {},
        fragment: ""
      }
      /* Array of text commands that need labelling */
      this.handleChange = this.handleChange.bind(this);
      this.updateTextValue = this.updateTextValue.bind(this);
      this.handleTextChange = this.handleTextChange.bind(this);
      this.handleNameChange = this.handleNameChange.bind(this);
      this.logSerialized = this.logSerialized.bind(this);
      this.componentDidMount = this.componentDidMount.bind(this);
      this.updateLabels = this.updateLabels.bind(this);
      this.selectCommand = this.selectCommand.bind(this);
      this.updateCommandWithSubstitution = this.updateCommandWithSubstitution.bind(this);
      this.uploadData = this.uploadData.bind(this);
      this.incrementIndex = this.incrementIndex.bind(this);
      this.decrementIndex = this.decrementIndex.bind(this);
    }
  
    componentDidMount() {
      fetch("http://localhost:9000/readAndSaveToFile/get_labels_progress")
        .then(res => res.json())
        .then((data) => { this.setState({ dataset: data }) })
        .then(() => console.log(this.state.dataset))
        .then(() => this.setState({ allCommands: Object.keys(this.state.dataset) }))
        .then(() => console.log(this.state.allCommands))

      fetch("http://localhost:9000/readAndSaveToFile/get_templates")
        .then(res => res.json())
        .then((data) => { this.setState({ templates: data }) })
        .then(() => console.log(this.state.templates))
    }

    handleTextChange(e) {
      this.setState({ command: e.target.value });
    }

    handleNameChange(e) {
      this.setState({ name: e.target.value });
    }
  
    writeLabels(data, is_template=false) {
      const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      };
      if (is_template) {
        fetch("http://localhost:9000/readAndSaveToFile/writeTemplates", requestOptions)
          .then(
            (result) => {
              console.log("success")
              console.log(result)
            },
            (error) => {
              console.log(error)
            }
          )
      } else {
        fetch("http://localhost:9000/readAndSaveToFile/writeLabels", requestOptions)
          .then(
            (result) => {
              console.log("success")
              console.log(result)
              this.setState({ value: "", command: "", name: "" })
              alert("saved!")
            },
            (error) => {
              console.log(error)
            }
          )
     }
    }
  
    handleChange(e) {
      this.setState({ value: e.target.value });
    }

    incrementIndex() {
      console.log("Moving to the next command")
      if (this.state.currIndex + 1 >= this.state.allCommands.length) {
        alert("Congrats! You have reached the end of annotations.")
      }
      this.setState({ name: this.state.allCommands[this.state.currIndex + 1], currIndex: this.state.currIndex + 1, value: JSON.stringify(this.state.dataset[this.state.allCommands[this.state.currIndex + 1]] ?? {}, undefined, 4) });
    }
  
    decrementIndex() {
      console.log("Moving to the next command")
      console.log(this.state.currIndex)
      this.setState({ name: this.state.allCommands[this.state.currIndex - 1], currIndex: this.state.currIndex - 1, value: JSON.stringify(this.state.dataset[this.state.allCommands[this.state.currIndex - 1]] ?? {}, undefined, 4) });
    }

    updateTextValue(text) {
      // Updates the logical form action dict value
      this.setState({ value: text });
    }

    updateAllCommands(text) {
      // Add a new command to the set of all commands
      console.log(text)
      let items = [...this.state.allCommands];
      items.push(text);
      this.setState({ allCommands: items }, function () {
        console.log("Updated list of commands")
      })
    }
  
  
    updateLabels(e) {
      // Make a shallow copy of the items
      try {
        // First check that the string is JSON valid
        let JSONActionDict = JSON.parse(this.state.value)
        let JSONString = {
          "command": this.state.command,
          "name": this.state.name,
          "logical_form": JSONActionDict
        }
        console.log(JSONString)
        let items = { ...this.state.dataset };
        items[this.state.name] = JSONString
        // Set state to the data items
        this.setState({ dataset: items }, function () {
          try {
            let actionDict = JSONActionDict
            console.log("writing dataset")
            console.log(JSONString)
            // save to disk
            this.writeLabels(this.state.dataset)
            // update the current commands
            this.updateAllCommands(this.state.command)
          } catch (error) {
            console.error(error)
            console.log("Error parsing JSON")
            alert("Error: Could not save logical form. Check that JSON is formatted correctly.")
          }
        });

        // Write templates
        // TODO: switch to name
        if (this.state.command !== "") {
          let templates = { ...this.state.templates };
          templates[this.state.name] = JSONString
                  // Set state to the data items
          this.setState({ templates: templates }, function () {
            try {
              let actionDict = JSONActionDict
              console.log("writing dataset")
              console.log(JSONString)
              // save to disk
              this.writeLabels(this.state.templates, true)
            } catch (error) {
              console.error(error)
              console.log("Error parsing JSON")
              alert("Error: Could not save logical form. Check that JSON is formatted correctly.")
            }
          });
        }
      } catch (error) {
        console.error(error)
        console.log("Error parsing JSON")
        alert("Error: Could not save logical form. Check that JSON is formatted correctly.")
      }
    }
  
    logSerialized() {
      console.log("saving serialized tree")
      // Save to local storage and disk
      this.updateLabels()
      // 
    }

    selectCommand(event, value) {
      // Update the current command selected and render the corresponding action dictionary
      if (value in this.state.dataset) {
        let selectedDict = this.state.dataset[value]
        let logical_form;
        let command;
        if ("logical_form" in selectedDict) {
          logical_form = selectedDict.logical_form
          command = selectedDict.command
        } else {
          logical_form = selectedDict
          command = ""
        }
        this.setState({ 
          command: command, 
          value: JSON.stringify(logical_form, undefined, 4),
          name:  value
        })
      } else {
        this.setState({ name: value, value: JSON.stringify({}) })
      }
    }

    updateCommandWithSubstitution(replaceText, substituteText) {
      // Update state for command
      console.log(substituteText)
      console.log(this.state.command)
      let newCommand = this.state.command.replace(replaceText, substituteText)
      console.log(newCommand)
      this.setState({ command: newCommand })
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
          <Autocomplete
            id="combo-box-demo"
            options={this.state.allCommands}
            getOptionLabel={(option) => option}
            getOptionSelected={(option, value) => option === value}
            style={{ width: 300 }}
            renderInput={(params) => <TextField {...params} label="Choose Command" variant="outlined" />}
            onChange={this.selectCommand}
          />
          <b> {this.props.title} </b>
          <ListComponent value={this.state.command} onChange={this.handleTextChange} />
          <div> Name of Template / Annotation </div>
          <ListComponent value={this.state.name} onChange={this.handleNameChange} />
          <div>
            <button style={{ marginBottom: 5, marginRight: 5, marginTop: 5 }} onClick={this.decrementIndex}>Prev</button>
            <button style={{ marginBottom: 5, marginRight: 5, marginTop: 5 }} onClick={this.incrementIndex}>Next</button>
          </div>
          <LogicalForm title="Action Dictionary" updateTextValue={this.updateTextValue} onChange={this.handleChange} updateCommand={(x, y) => this.updateCommandWithSubstitution(x, y)} currIndex={this.state.fragmentsIndex} value={this.state.value} schema={this.props.schema} dataset={this.state.dataset} />
          <div onClick={this.logSerialized}>
            <button>Save</button>
          </div>
          <div style={{ padding: 10 }} onClick={this.uploadData}>
            <button>Create Dataset</button>
          </div>
        </div>
      )
    }
  }

  export default TemplateAnnotator;