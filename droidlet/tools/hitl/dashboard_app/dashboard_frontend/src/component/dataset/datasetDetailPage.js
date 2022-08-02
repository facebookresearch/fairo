/*
Copyright (c) Facebook, Inc. and its affiliates.

The detail page for viewing a dataset.
Takes optional state parameter from react-router-dom navigation (navigate, link, navlink ... etc), the optipnal states are:
- datasetList: a list of dataset of current pipeline

Usage:
<DatasetDetailPage />
*/
import { Card, List, Select, Spin, Tooltip, Typography } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import { SocketContext } from "../../context/socket";

const { Option } = Select;

const DatasetJSONCmp = (props) => {
    const obj = props.obj;
    const [jsonStr, setJsonStr] = useState(props.jsonStr);
    const [tabSep, setTabSep] = useState(false);

    const handleOnClickJson = () => {
        // change json format
        if (!tabSep) {
            setTabSep(true);
            const newJsonStr = JSON.stringify(obj, null, 2);
            setJsonStr(newJsonStr);
        } else {
            setTabSep(false);
            const newJsonStr = JSON.stringify(obj);
            setJsonStr(newJsonStr);
        }
    }

    useEffect(() => { }, [jsonStr]);

    return <pre onClick={() => handleOnClickJson()} >
        <Tooltip title= {tabSep ? "Collapse" : "Expand"}>
            <Typography.Text>
                {jsonStr}
            </Typography.Text>
        </Tooltip>

    </pre>
}

const DatasetDetailPage = (props) => {
    const socket = useContext(SocketContext);
    const location = useLocation();
    const pipelineType = useParams().pipeline;
    const [datasetList, setDatasetList] = useState((location.state && location.state.datasetList) ? location.state.datasetList : []);
    const [datasetContent, setDatasetContent] = useState(null);
    const [selectedDatasetVer, setSelectedDatasetVer] = useState(null);
    const onSelectDatasetVer = (value) => {
        setSelectedDatasetVer(value);
    }

    const handleReceivedDatasetList = useCallback((data) => {
        setDatasetList(data.sort().reverse());
    }, []);



    const handleRecievedDatasetContent = useCallback((data) => {
        const formattedData = data.split("\n").map((line) => {
            line = line.split("|");
            line[2] = line[2] && line[2].replace("\\", "");
            try {
                line[2] = JSON.parse(line[2]);
            } catch (e) {
                // do nothing if it is undefined (encounter parse error)
            }
            line[2] = { obj: line[2], jsonStr: JSON.stringify(line[2])};
            return line;
        });

        setDatasetContent(formattedData);
    }, []);

    const getDatasetList = () => {
        socket.emit("get_dataset_list_by_pipeleine", pipelineType);
    }

    useEffect(() => {
        socket.on("get_dataset_list_by_pipeleine", (data) => handleReceivedDatasetList(data));
        socket.on("get_dataset_by_name", (data) => handleRecievedDatasetContent(data));
    }, [socket, handleReceivedDatasetList, handleRecievedDatasetContent]);

    useEffect(() => {
        datasetList.length === 0 && getDatasetList();
    }, []); // component did mount, reterive dataset list if not passed in

    useEffect(() => {
        // set default dataset version when dataset list is reterived
        datasetList.length && setSelectedDatasetVer(datasetList[0]);
    }, [datasetList]);

    useEffect(() => {
        // get dataset content when a dataser version is selected
        setDatasetContent(null);
        selectedDatasetVer && socket.emit("get_dataset_by_name", selectedDatasetVer);
    }, [selectedDatasetVer]);

    return <div>
        <Typography.Title level={4}>
            Datasets For {pipelineType} Pipeline
        </Typography.Title>
        <div style={{ display: 'flex', paddingLeft: '24px' }}>
            <div style={{ width: '20%' }}>
                <Typography.Title level={5}>
                    View version
                </Typography.Title>
                {/* dataset list here */
                    datasetList.length ?
                        <div>
                            <Select
                                style={{ width: '240px' }}
                                showSearch
                                defaultValue={datasetList[0]}
                                placeholder="Select a version"
                                optionFilterProp="children"
                                onChange={onSelectDatasetVer}
                                filterOption={(input, option) => option.children.toLowerCase().includes(input.toLowerCase())}
                            >

                                {datasetList.map((datasetName) => (
                                    <Option value={datasetName}>
                                        {datasetName}
                                    </Option>
                                ))}
                            </Select>
                        </div> :
                        <Spin />
                }
            </div>
            <div style={{ padding: '0 18px 0 36px', width: '80%' }}>
                {/* dataset detail  */}
                <Card loading={!datasetContent}>
                    <List style={{ overflow: 'auto', height: '80vh', textAlign: "left" }}>
                        {
                            datasetContent
                            && datasetContent.map((line, idx) =>
                                <List.Item>
                                    <List.Item.Meta
                                        title={<div>
                                            <Typography.Text strong>{"[" + line[0] + "]"}</Typography.Text>
                                            <Typography.Text style={{ paddingLeft: "6px" }}>{line[1]}</Typography.Text>
                                        </div>}
                                        description={
                                            <DatasetJSONCmp obj={line[2].obj} jsonStr={line[2].jsonStr} />
                                        }
                                    />
                                </List.Item>)
                        }
                    </List>
                </Card>
            </div>
        </div>
    </div>;
}

export default DatasetDetailPage;
