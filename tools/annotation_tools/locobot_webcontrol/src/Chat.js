/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import './Chat.css'

class Chat extends React.Component{

    constructor(props){
        super(props)

        this.submit = this.submit.bind(this)
    }

    render(){
        let chat_history = this.props.history.map((item, key) =>
            <p>{item}</p>
        )

        return(
            <div class="message-container">
                <div class="messages" id="messages">
                    {chat_history}
                    <div class="scrollfix"></div>
                    <div class="scrollfix"></div>
                    <div class="scrollfix"></div>
                </div>
                <div class="message-bar">
                    <input id="msg-box" class="text-input" placeholder="Message" onKeyPress={this.submit} type="text"></input>
                </div>
                
            </div>
        )
    }

    scroll(){
        let messageBox = document.getElementById("messages")
        messageBox.scrollTop = messageBox.scrollHeight
    }

    submit(e){
        if(e.key == "Enter"){
            let inputBox = document.getElementById("msg-box")
            this.props.msgCallback(inputBox.value)
            inputBox.value = ""

            this.scroll()

        }
        
    }
}

export default Chat