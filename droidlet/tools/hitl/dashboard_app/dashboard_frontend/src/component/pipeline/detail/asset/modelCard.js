/*
Copyright (c) Facebook, Inc. and its affiliates.

The card showing Model infomation of a run. 
**Note: this is just a placeholder for the model card. TODO: add real content of the model card**

Usage:
<ModelCard />
*/
import { Card, Descriptions } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";

const { Meta } = Card;

const ModelCard = (props) => {
    const batchId = props.batchId;
    const pipelineType = props.pipelineType;
    const [modelArgs, setModelArgs] = useState(null);
    const [loading, setLoading] = useState(true);

    const socket = useContext(SocketContext);

    const handleReceivedModelArgs = (data) => {
        setLoading(false);
        console.log(typeof (JSON.parse(data)));
        console.log(Object.keys(JSON.parse(data)))

        if (data !== 404) {
            setModelArgs(JSON.parse(data));
        }
    }

    const getModelArgs = () => {
        socket.emit("get_model_value_by_id_n_key", batchId, "args");
    }

    useEffect(() => {
        socket.on("get_model_value_by_id_n_key", (data) => handleReceivedModelArgs(data));
    }, [socket, handleReceivedModelArgs]);

    useEffect(() => {
        getModelArgs();
    }, []); // component did mount

    const processModelArg = (arg) => {
        if (typeof(arg) === "object") {
            return Object.keys(arg).map((key) => (`${key}: ${arg[key]}`));
        } else if (!arg || (typeof(arg) === "string" && arg.length === 0)){ 
            return "NA";
        } else{
            return arg;
        } 
    }

    return (
        <div style={{ width: '70%' }}>
            <Card title="Model" loading={loading}>
                <Meta />
                <Descriptions title="Model Args" bordered column={2}>
                    {modelArgs && Object.keys(modelArgs).map((key) =>
                        <Descriptions.Item label={key}>{processModelArg(modelArgs[key])}</Descriptions.Item>
                    )}
                </Descriptions>

            </Card>
        </div>);
}

export default ModelCard;