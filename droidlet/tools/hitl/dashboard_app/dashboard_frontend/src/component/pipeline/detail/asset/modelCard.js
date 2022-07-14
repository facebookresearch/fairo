/*
Copyright (c) Facebook, Inc. and its affiliates.

The card showing Model infomation of a run. 
**Note: this is just a placeholder for the model card. TODO: add real content of the model card**

Usage:
<ModelCard />
*/
import { Button, Card, Descriptions, Divider, Tooltip } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";
import ModelAtrributeModal from "./modelAttributeDetailModal";

const { Meta } = Card;

const ModelCard = (props) => {
    const batchId = props.batchId;
    const pipelineType = props.pipelineType;
    const [modelArgs, setModelArgs] = useState(null);
    const [modelKeys, setModelKeys] = useState(null);
    const [loadingArgs, setLoadingArgs] = useState(true);
    const [loadingKeys, setLoadingKeys] = useState(true);
    const [currentModelKey, setCurrentModelKey] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);

    const socket = useContext(SocketContext);

    const handleReceivedModelArgs = useCallback((data) => {
        setLoadingArgs(false);
        console.log(data)
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

    const getModelArgs = () => {
        socket.emit("get_model_value_by_id_n_key", batchId, "args");
    }

    const getModelKeys = () => {
        socket.emit("get_model_keys_by_id", batchId);
    }

    useEffect(() => {
        socket.on("get_model_value_by_id_n_key", (data) => handleReceivedModelArgs(data));
        socket.on("get_model_keys_by_id", (data) => handleRecievedModelKeys(data));
    }, [socket, handleReceivedModelArgs]);

    useEffect(() => {
        loadingArgs && getModelArgs();
        loadingKeys && getModelKeys();
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
        console.log(modelKey);
        setCurrentModelKey(modelKey);
        setModalOpen(true);
    }

    return (
        <div style={{ width: '70%' }}>
            <Card title="Model" loading={loadingKeys || loadingArgs}>
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
                            </div>
                            :
                            <div>NA</div>
                    )
                }
                {/* modal showing a specific model attribute's field*/}
                {currentModelKey
                    &&
                    <ModelAtrributeModal
                        modelKey={currentModelKey}
                        setModelKey={setCurrentModelKey}
                        modalOpen={modalOpen}
                        setModalOpen={setModalOpen}
                        batchId={batchId} 
                    />
                }
            </Card>

        </div>);
}

export default ModelCard;