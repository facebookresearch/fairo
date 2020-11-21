/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import Instructions from './Instructions';
import ObjectCorrection from './prebuilt-components/ObjectCorrection';

const INSTRUCTIONS = `
Welcome to our task, and thank your for accepting this HIT
Please read these instructions carefully or your HIT will be rejected
-Your goal is to correct any errors found in the data
-You will be given a red box around a portion of an image
-Verify and correct the name of the outlined item, and it's properties
-Outline the object by following the onscreen instructions
-The task will auto-submit when completed
`

class App extends React.Component {
  constructor(props){
    super(props)

    this.instructionsShowing = false
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
          <ObjectCorrection imgUrl={window.ANN_URL} targets={window.ANN_TARGETS}/>
        </div>
    )}}

    
}

export default App;
