/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import AppBar from 'material-ui-core/AppBar';
import Tabs from 'material-ui-core/Tabs';
import Tab from 'material-ui-core/Tab';
import Chat from './Chat'
import Remote from './Remote'

function a11yProps(index) {
    return {
      id: `full-width-tab-${index}`,
      'aria-controls': `full-width-tabpanel-${index}`,
    };
  }

class LocobotControl extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            tabValue: 0,
            messageHistory: ["Welcome! Send commands to your bot below", "Just type a message and press ENTER"]
        }

        this.paneSelector = this.paneSelector.bind(this)
        this.tabClicked = this.tabClicked.bind(this)
        this.addMessage = this.addMessage.bind(this)
        this.socket = this.props.socket
        this.socket.on('reply', data => {
            this.addMessage(data, true)
        })
    }
 
    //Rendering

    paneSelector() {
        console.log(this.tabValue)
        if (this.state.tabValue == 0){
            return <Chat history={this.state.messageHistory} msgCallback={this.addMessage}/>
        } else {
            return <Remote></Remote>
        }
    }

    render() {
        return(
            <div class="control-panel">
                <AppBar position="static" color="default">
                <Tabs
                    value={this.state.tabValue}
                    onChange={this.tabClicked}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="fullWidth"
                    aria-label="full width tabs example"
                >
                    <Tab label="Text Console" {...a11yProps(0)} />
                    <Tab label="Manual" {...a11yProps(1)} />
                </Tabs>
                </AppBar>
                {this.paneSelector()}
            </div>
        )
    }

    //Interactions

    tabClicked(e, val) {
        this.setState({tabValue: val})
    }

    addMessage(msg, isReply=false){
        this.setState(prevState => ({
            messageHistory: [...prevState.messageHistory, msg]
          }))
        if( !isReply ){
            this.socket.send(msg)
        }
    }

}

export default LocobotControl