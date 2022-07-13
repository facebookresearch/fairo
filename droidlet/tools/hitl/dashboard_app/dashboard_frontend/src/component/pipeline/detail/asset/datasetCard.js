/*
Copyright (c) Facebook, Inc. and its affiliates.

The card showing dataset infomation of a run. 
Takes batchid and pipelineType as input. 

Usage:
<DatasetCard batchId = {batchId} pipelineType = {pipelineType} />
*/

import { Card, Descriptions, Tooltip, Typography } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import { Link } from 'react-router-dom';
import { SocketContext } from "../../../../context/socket";

const { Meta } = Card;

const DatasetCard = (props) => {
    const socket = useContext(SocketContext);
    const batchId = props.batchId;
    const pipelineType = props.pipelineType;
    const [loading, setLoading] = useState(true);
    const [datasetIndecies, setDatasetIndecies] = useState([]);
    const [datasetList, setDatasetList] = useState([]);

    const handleRecievedDatasetIndecies = useCallback((data) => {
        if (data === 404) {
            // received error code, set indecies to NA
            setDatasetIndecies(["NA", "NA"]);
        }
        setDatasetIndecies(data);
    }, []);

    const getDatasetIndecies = () => {
        socket.emit("get_dataset_idx_by_id", batchId);
    }

    const handleReceivedDatasetList = useCallback((data) => {
        setDatasetList(data.sort().reverse());
    }, []);

    const getDatasetList = () => {
        socket.emit("get_dataset_list_by_pipeleine", pipelineType);
    }

    useEffect(() => {
        socket.on("get_dataset_idx_by_id", (data) => handleRecievedDatasetIndecies(data));
        socket.on("get_dataset_list_by_pipeleine", (data) => handleReceivedDatasetList(data));
    }, [socket, handleRecievedDatasetIndecies, handleReceivedDatasetList]);


    useEffect(() => {
        getDatasetIndecies();
        getDatasetList();
    }, []); // component did mount

    useEffect(() => {
        setLoading(datasetList.length === 0 || datasetIndecies.length === 0);
    }, [datasetIndecies, datasetList]); // update comp

    return (
        <div style={{ paddingRight: '16px', width: '50%' }}>
            <Card title="Dataset" loading={loading}>
                <Meta />
                <Descriptions bordered>
                    <Descriptions.Item label="Data Indecies" span={3}>
                        {datasetIndecies[0]} - {datasetIndecies[1]}
                    </Descriptions.Item>
                    <Descriptions.Item label="Dataset" span={3}>
                        <Tooltip title="View Dataset">
                            <Typography.Link >
                                <Link
                                    to={`/dataset/${pipelineType}`}
                                    state={{ datasetList: datasetList }}
                                    replace={true}
                                >
                                    {datasetList[0]}
                                </Link>
                            </Typography.Link>
                        </Tooltip>
                    </Descriptions.Item>
                </Descriptions>
            </Card>
        </div>);
}

export default DatasetCard;
