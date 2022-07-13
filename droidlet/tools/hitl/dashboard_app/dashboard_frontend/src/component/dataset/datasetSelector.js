/*
Copyright (c) Facebook, Inc. and its affiliates.

Dataset selector used for selecting a specific pipeline and navigate to the pipeline's dataset detail page.

Usage:
<DatasetSelector />
*/
import { Select, Typography } from "antd";
import React from "react";
import { useNavigate } from "react-router-dom";
import PIPELINE_TYPES from "../../constants/datasetContants";

const { Option } = Select;

const DatasetSelector = () => {
    const navigate = useNavigate();
    const onSelectingPipeline = (selectedValue) => {
        navigate(selectedValue);
    };
    
    return <div>
        <Typography.Title
            level={5}
            style={{ paddingBottom: '8px' }}>
            Please select a pipeline to start.
        </Typography.Title>
        <Select
            style={{ width: '240px' }}
            showSearch
            placeholder="Select a pipeline"
            optionFilterProp="children"
            onChange={onSelectingPipeline}
            filterOption={(input, option) => option.children.toLowerCase().includes(input.toLowerCase())}
        >

            {PIPELINE_TYPES.map((pipelineType) => (
                <Option value={pipelineType}>
                    {pipelineType}
                </Option>
            ))}
        </Select>
    </div>;
}

export default DatasetSelector;