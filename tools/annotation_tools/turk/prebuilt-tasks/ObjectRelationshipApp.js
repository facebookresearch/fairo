/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import Instructions from './Instructions';
import RelationshipAnnotator from './prebuilt-components/ObjectRealtionships';

const INSTRUCTIONS = `
Welcome to our task, and thank your for accepting this HIT
Please read these instructions carefully or your HIT will be rejected
-Your goal is to write sentances describing parts of the image
-Write as many sentances as possible (press Enter after each sentance)
-After you finish writing, you will then draw boxes around the parts of the image described
-View the examples on the next page before writing, or else your task will be rejected
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
          <RelationshipAnnotator
            imgUrl={window.ANN_URL}
          />
        </div>
    )}}

    
}

export default App;
