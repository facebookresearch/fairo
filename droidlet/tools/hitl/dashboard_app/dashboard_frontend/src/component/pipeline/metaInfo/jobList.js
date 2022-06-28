import React from 'react';

const JobList = (props) => {
    const pipelineType = props.pipelineType;

    return (
        <div>
            {pipelineType} job list
        </div> 
    );
}

export default JobList;