import React from 'react';

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
          <b> {this.props.title}</b>
          <textarea rows="20" cols="100" value={this.props.value} onKeyDown={this.keyPress} onChange={(e) => this.props.onChange(e)} fullWidth={false} />
        </div>
      )
    }
  }

  export default LogicalForm;