/*
Copyright (c) Facebook, Inc. and its affiliates.

The card showing Model infomation of a run. 
**Note: this is just a placeholder for the model card. TODO: add real content of the model card**

Usage:
<ModelCard />
*/
import { Card } from "antd";
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
        console.log(data)
        if (data !== 404) {
            setModelArgs(data);
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

    return (
        <div style={{ width: '70%' }}>
            <Card title="Model" loading={loading}>
                <Meta />
                {modelArgs}
            </Card>
        </div>);
}

export default ModelCard;