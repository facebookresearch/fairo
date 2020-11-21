/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import Instructions from './Instructions';

// Add instructions for the task here (seperate by newlines)
const INSTRUCTIONS = `
Welcome to our task, and thank your for accepting this HIT
Please read these instructions carefully or your HIT will be rejected
-Example instruction
`

class App extends React.Component {
  constructor(props){
    super(props)

    this.instructionsShowing = true
  }

  render(){ 
    if(this.instructionsShowing){
      
      return(
        <Instructions
          instructions={INSTRUCTIONS}
          minutes={window.MINUTES}
          finishedCallback={()=>{
            this.instructionsShowing = false
            this.forceUpdate()
          }}
        ></Instructions>
        )
    
    } else {
      
      return(
        <div>
          <p>YOUR REACT APP ROOT COMPONENT HERE</p>
        </div>
    )}}

    
}

export default App;
