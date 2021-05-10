import React from 'react';

// Represents a Text Input node
class LogicalForm extends React.Component {
  constructor(props) {
    super(props)
    this.keyPress = this.keyPress.bind(this)
    this.parseTemplates = this.parseTemplates.bind(this)
    this.state = {
      fragment: ""
    }
  }

  parseTemplates(actionDict) {
    var keys = Object.keys(actionDict)
    for (var i = 0; i < keys.length; i++) {
      if (Array.isArray(actionDict[keys[i]])) {
        // Currently grabbing the first, to change to check all
        this.parseTemplates(actionDict[keys[i]][0])
      } else if (typeof (actionDict[keys[i]]) == "object") {
        this.parseTemplates(actionDict[keys[i]])
      } else if (keys[i].match(/<.+>/g) && actionDict[keys[i]].match(/<.+>/g)) {
        // The key that we are substituting
        var matchKey = /<(.+)>/g.exec(keys[i])[1]
        let substituteText = /<(.+)>/g.exec(actionDict[keys[i]])[1]
        // TODO: don't iterate over full dataset, only fragments
        let commands = Object.keys(this.props.dataset)
        for (var j = 0; j < commands.length; j++) {
          if (substituteText !== commands[j]) {
            continue;
          }
          let actionDictObj = this.props.dataset[commands[j]]
          let logicalForm;
          // Template objects
          if (Object.keys(actionDictObj).includes("logical_form")) {
            logicalForm = actionDictObj["logical_form"]
            let text = actionDictObj["command"]
            // Check that the logical form contains the key we are inserting into
            // NOTE: assuming a top level key matches up here
            if (Object.keys(logicalForm).includes(matchKey)) {
              console.log(`Found substitution for key ${matchKey}`)
              // Update the action dictionary with the inserted logical form
              actionDict[keys[i]] = logicalForm[matchKey]
              // Update the command field with the text substitution
              this.props.updateCommand(keys[i], text)
            }
          }
        }
      }
    }
  }

  keyPress(e) {
    // Hit enter
    if (e.keyCode == 13) {
      e.preventDefault()
      let text = ""
      try {
        let autocompletedResult = e.target.value
        // Apply replacements
        let definitions = Object.keys(this.props.schema)
        definitions.forEach(node => {
          let node_def = this.props.schema[node]
          let properties_subtree = {}
          // Template objects
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
          } else if (node_def["type"] == "array" && "properties" in node_def["items"]) {
            let node_properties = Object.keys(node_def["items"]["properties"])
            node_properties.forEach(key => {
              properties_subtree[key] = ""
            })
          }
          // if the keys include dialogue type, replace the whole path and insert dialogue type
          // console.log(node + ": " + JSON.stringify(properties_subtree))
          if ("dialogue_type" in properties_subtree) {
            properties_subtree["dialogue_type"] = node.toUpperCase()
            autocompletedResult = autocompletedResult.replace('"' + node + '"' + ":  ", JSON.stringify(properties_subtree))
          } else {
            autocompletedResult = autocompletedResult.replace('"' + node + '"' + ":  ", '"' + node + '"' + ": " + JSON.stringify(properties_subtree))
          }
        }
        )
        // Insert fragments and lists
        // let commands = Object.keys(this.props.dataset)
        // for (var i = 0; i < commands.length; i++) {
        //     let actionDictObj = this.props.dataset[commands[i]]
        //     let actionDict;
        //     // Template objects
        //     if (!autocompletedResult.includes(commands[i])) {
        //       continue
        //     }
        //     if (Object.keys(actionDictObj).includes("logical_form")) {
        //       actionDict = actionDictObj["logical_form"]
        //       text = actionDictObj["command"]
        //       console.log(actionDict)
        //       console.log("Found template object")
        //       this.props.updateCommand("<location>", text)
        //       // Update the command field with the text substitution
        //     } else if (Array.isArray(actionDictObj)) {
        //         // If the substitution is a list of subtrees, pick a random one
        //         actionDict = actionDictObj[Math.floor(Math.random() * actionDictObj.length)]
        //     } else {
        //         actionDict = actionDictObj
        //     }
        //     autocompletedResult = autocompletedResult.replace('<' + commands[i] + '>', JSON.stringify(actionDict))
        // }
        // Recursive updates
        // Find the template objects
        var actionDict = JSON.parse(autocompletedResult);
        this.parseTemplates(actionDict);
        // Apply JSON formatting       
        // console.log(JSON.stringify(autocompletedResult))
        var pretty = JSON.stringify(actionDict, undefined, 4);
        this.props.updateTextValue(pretty)
        e.target.value = pretty
      }
      catch (err) {
        console.log(err)
      }
    }
  }

  render() {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 10, marginTop: 10 }} >
        <b> {this.props.title}</b>
        <textarea rows="20" cols="100" value={this.props.value} onKeyDown={this.keyPress} onChange={(text) => this.props.onChange(text)} fullWidth={false} />
      </div>
    )
  }
}

export default LogicalForm;