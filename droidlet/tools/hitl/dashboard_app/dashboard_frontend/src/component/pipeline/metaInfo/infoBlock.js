import React from 'react';

const InfoBlock = (props) => {
    const infoType = props.infoType;
    const pipelineType = props.pipelineType;

    return (
        <div>
            {infoType} : {pipelineType.label}
        </div> 
    );
}

export default InfoBlock;