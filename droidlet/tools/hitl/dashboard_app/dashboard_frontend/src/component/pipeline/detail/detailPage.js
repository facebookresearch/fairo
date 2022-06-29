/*
Copyright (c) Facebook, Inc. and its affiliates.

Detail page of a run.

Batach ID and pipeline type needs to be specified by the caller. 

Usage:
<DetailPage batchId={batchId} pipelineType={pipelineType} />
*/
import React from "react";

const DetailPage = (props) => {
    const batchId = props.batchId;
    const pipelineType = props.pipelineType;

    return <div> Place Holder ... {batchId} {pipelineType.label}</div>;

}

export default DetailPage;