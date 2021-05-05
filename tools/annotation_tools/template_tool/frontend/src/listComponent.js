import React from 'react';

// Represents a Text Input node
class ListComponent extends React.Component {
    constructor(props) {
      super(props)
    }
  
    render() {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 5, marginTop: 5 }}>
          <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 5, marginTop: 5 }}>
          <textarea rows="2" cols="10" value={this.props.value} onChange={(e) => this.props.onChange(e)} fullWidth={false} />
          </div>
        </div>
      )
    }
  }

export default ListComponent;