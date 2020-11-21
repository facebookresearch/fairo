/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';

class Video extends React.Component {
    constructor(props){
        super(props)
    }

    render(){
        return(
            <img src={this.props.ip + "/video_feed"}></img>
        )
    }
}

export default Video