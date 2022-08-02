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
    const [datasetIndicies, setDatasetIndicies] = useState([]);
    const [datasetList, setDatasetList] = useState([]);

    const handleRecievedDatasetIndicies = useCallback((data) => {
        if (data === 404) {
            // received error code, set indices to NA
            setDatasetIndicies(["NA"]);
        }
        setDatasetIndicies(data);
    }, []);

    const getDatasetIndicies = () => {
        socket.emit("get_dataset_idx_by_id", batchId);
    }

    const handleReceivedDatasetList = useCallback((data) => {
        setDatasetList(data.sort().reverse());
    }, []);

    const getDatasetList = () => {
        socket.emit("get_dataset_list_by_pipeleine", pipelineType);
    }

    useEffect(() => {
        socket.on("get_dataset_idx_by_id", (data) => handleRecievedDatasetIndicies(data));
        socket.on("get_dataset_list_by_pipeleine", (data) => handleReceivedDatasetList(data));
    }, [socket, handleRecievedDatasetIndicies, handleReceivedDatasetList]);


    useEffect(() => {
        getDatasetIndicies();
        getDatasetList();
    }, []); // component did mount

    useEffect(() => {
        setLoading(datasetList.length === 0 || datasetIndicies.length === 0);
    }, [datasetIndicies, datasetList]); // update comp

    return (
        <div style={{ paddingRight: '16px', width: '30%' }}>
            <Card title="Dataset" loading={loading}>
                <Meta />
                <Descriptions bordered>
                    <Descriptions.Item label="Data Indicies" span={3}>
                        {datasetIndicies.length === 2 ? `${datasetIndicies[0]} - ${datasetIndicies[1]}` : "NA"}
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
