/*
Copyright (c) Facebook, Inc. and its affiliates.

Meta Info Block for a pipeline. 
InfoType and pipelineType need to be specified by the caller.

To use this component:
<InfoBlock infoType={infoType} pipelineType={pipelineType} />
*/
import React from 'react';

const InfoBlock = (props) => {
    const infoType = props.infoType;
    const pipelineType = props.pipelineType;

    return (
        <div>
            Placeholder ... {infoType} : {pipelineType.label}
        </div> 
    );
}

export default InfoBlock;