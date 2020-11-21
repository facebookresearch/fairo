/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import ObjectAnnotator from './prebuilt-components/ObjectAnnotation'
import Instructions from './Instructions';

const INSTRUCTIONS = `
Welcome to our task, and thank your for accepting this HIT
Please read these instructions carefully or your HIT will be rejected
-Your goal is to label and outline as many objects as possible
-If part of an object is blocked, outline the visible parts only
-Stay as close as possible to the object, or else your HIT may be rejected
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
          <ObjectAnnotator imgUrl={window.ANN_URL}/>
        </div>
    )}}

    
}

export default App;
