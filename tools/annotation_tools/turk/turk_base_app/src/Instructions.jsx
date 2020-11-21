/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react'

/*Props
instrucitons: String
    Instructions to display

finishedCallback: function
    Function to call once the instrutions have been read

*/

class Instructions extends React.Component{

    render(){
        return(
            <div>
                {this.props.instructions.split('\n').map((line)=>{
                    return <p>{line}</p>
                })}
                <p>You have {this.props.minutes} minutes to complete this task</p>
                <button onClick={this.props.finishedCallback}>I Understand</button>
            </div>
        )
    }
}

export default Instructions