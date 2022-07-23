/*
Copyright (c) Facebook, Inc. and its affiliates.

The card showing Model infomation of a run. 

Usage:
<ModelCard batchId = {batchId} pipelineType={pipelineType}/>
*/
import { Button, Card, Descriptions, Divider, Tooltip, Typography } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";
import ModelAtrributeModal from "./modelAttributeDetailModal";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, Tooltip as ChartTooltip } from 'recharts';

const { Meta } = Card;
const LOSS_ACC_TYPES = [{ "label": "Epoch", "value": "epoch" }, { "label": "Text Span", "value": "text_span" }]

const ModelLossAccGraph = (props) => {
    const data = props.data.map((o, idx) => ({ Loss: o.loss, Accuracy: o.acc, Epoch: idx }));

    return <div style={{ width: "100%", height: "100%" }}>
        <LineChart
            width={600}
            height={400}
            data={data}
            margin={{
                top: 5,
                right: 30,
                left: 20,
                bottom: 5,
              }}
        >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="Epoch" label={{offset: 0, value: "Epoch", position: "insideBottom" }}/>
            <YAxis />
            <Legend />
            <ChartTooltip />
            <Line type="monotone" dataKey="Loss" stroke="#ad2102" activeDot={{ r: 8 }} />
            <Line type="monotone" dataKey="Accuracy" stroke="#1890ff" />
        </LineChart>
    </div>
}

const ViewLossAccCard = (props) => {
    const lossAccData = props.data;
    const [activeTabKey, setActiveTabKey] = useState(LOSS_ACC_TYPES[0]["value"]);

    return <Card
        tabList={LOSS_ACC_TYPES.map((o) => ({ tab: o["label"], key: o["value"] }))}
        activeTabKey={activeTabKey}
        onTabChange={(key) => setActiveTabKey(key)}
    >
        <ModelLossAccGraph data={lossAccData[activeTabKey]} />
    </Card>
}

const ModelCard = (props) => {
    const batchId = props.batchId;
    const pipelineType = props.pipelineType;
    const [modelArgs, setModelArgs] = useState(null);
    const [modelKeys, setModelKeys] = useState(null);
    const [loadingArgs, setLoadingArgs] = useState(true);
    const [loadingKeys, setLoadingKeys] = useState(true);
    const [currentModelKey, setCurrentModelKey] = useState(null);
    const [attrModalOpen, setAttrModalOpen] = useState(false);
    const [lossAccData, setLossAccData] = useState(null);
    const [loadingLossAcc, setLoadingLossAcc] = useState(true);

    const socket = useContext(SocketContext);

    const handleReceivedModelArgs = useCallback((data) => {
        setLoadingArgs(false);
        // data other than args are rendered in the ModelAttributeDetailModal component
        if (data !== 404 && data[0] === "args") {
            setModelArgs(JSON.parse(data[1]));
        }
    });

    const handleRecievedModelKeys = useCallback((data) => {
        setLoadingKeys(false);
        if (data !== 404) {
            setModelKeys(data);
        }
    });

    const handleReceivedLossAcc = useCallback((data) => {
        setLoadingLossAcc(false);
        if (data !== 404) {
            setLossAccData(data);
        }
    });

    const getModelArgs = () => {
        // get args for the model
        socket.emit("get_model_value_by_id_n_key", batchId, "args");
    }

    const getModelKeys = () => {
        socket.emit("get_model_keys_by_id", batchId);
    }

    const getLossAcc = () => {
        socket.emit("get_best_model_loss_acc_by_id", batchId);
    }

    useEffect(() => {
        socket.on("get_model_value_by_id_n_key", (data) => handleReceivedModelArgs(data));
        socket.on("get_model_keys_by_id", (data) => handleRecievedModelKeys(data));
        socket.on("get_best_model_loss_acc_by_id", (data) => handleReceivedLossAcc(data));
    }, [socket, handleReceivedModelArgs]);

    useEffect(() => {
        loadingArgs && getModelArgs();
        loadingKeys && getModelKeys();
        loadingLossAcc && getLossAcc();
    }, []); // component did mount

    const processModelArg = (arg) => {
        if (typeof (arg) === "object") {
            return Object.keys(arg).map((key) => (`${key}: ${arg[key]}`));
        } else if (!arg || (typeof (arg) === "string" && arg.length === 0)) {
            return "NA";
        } else {
            return arg;
        }
    }

    const handleOnClickViewModelAttibute = (modelKey) => {
        setCurrentModelKey(modelKey);
        setAttrModalOpen(true);
    }

    const handleViewModelLossAndAcc = (lossAccType) => {
        alert(lossAccData[lossAccType].map((o) => (`loss: ${o.loss},  acc: ${o.acc}`)));
    }

    return (
        <div style={{ width: '70%' }}>
            <Card title="Model" loading={loadingKeys || loadingArgs || loadingLossAcc}>
                <Meta />
                {
                    !loadingKeys && !loadingArgs && (
                        modelKeys ?
                            <div style={{ textAlign: "left" }}>
                                <Descriptions title="Model Args" bordered column={2}>
                                    {modelArgs && Object.keys(modelArgs).map((key) =>
                                        <Descriptions.Item label={key}>{processModelArg(modelArgs[key])}</Descriptions.Item>
                                    )}
                                </Descriptions>
                                <Divider />
                                <Descriptions title="Other Attributes" bordered column={2}>
                                    {modelKeys && modelKeys.map((modelKey) =>
                                        modelKey !== "args"
                                        &&
                                        <Descriptions.Item>
                                            <Tooltip title={`View ${modelKey}`}>
                                                <Button type="link" onClick={() => handleOnClickViewModelAttibute(modelKey)}>
                                                    {modelKey}
                                                </Button>
                                            </Tooltip>
                                        </Descriptions.Item>
                                    )}
                                </Descriptions>
                                {
                                    lossAccData &&
                                    <>
                                        <Divider />
                                        <Typography.Title level={5} style={{ paddingBottom: '18px' }}>Model Loss And Accuracy</Typography.Title>
                                        {/* {LOSS_ACC_TYPES.map((o) => (
                                                <Descriptions.Item>
                                                    <Tooltip title={`View ${o.label}`}>
                                                        <Button type="link" onClick={() => handleViewModelLossAndAcc(o.value)}>
                                                            {o.label}
                                                        </Button>
                                                    </Tooltip>
                                                </Descriptions.Item>
                                            ))} */}
                                        <ViewLossAccCard data={lossAccData} />
                                    </>
                                }
                            </div>
                            :
                            <div>NA</div>
                    )
                }
                {/* modal showing a specific model attribute's field (anything other than args) */}
                {currentModelKey
                    &&
                    <ModelAtrributeModal
                        modelKey={currentModelKey}
                        setModelKey={setCurrentModelKey}
                        modalOpen={attrModalOpen}
                        setModalOpen={setAttrModalOpen}
                        batchId={batchId}
                    />
                }
            </Card>

        </div>);
}

export default ModelCard;