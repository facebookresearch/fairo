/*
Copyright (c) Facebook, Inc. and its affiliates.

The modal showing a model's value of an attribute (modelKey). 
Used by the ModalCard component.

Usage:
<ModelAtrributeModal 
    batchId={batchId} 
    modelKey={modelKey}
    setModelKey={setModelKey}
    modalOpen={modalOpen}
    setModalOpen={setModalOpen}
/>
Note that modelKey and modalOpen must be state varible from the ModalCard component, 
otherwise the modal cannot be destoyed properly.
*/
import { Modal, Spin } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";

const ModelAtrributeModal = (props) => {
    const socket = useContext(SocketContext);
    const batchId = props.batchId;
    const modelKey = props.modelKey; // must be state from parent
    const setModelKey = props.setModelKey;
    const modalOpen = props.modalOpen; // must be state from parent
    const setModalOpen = props.setModalOpen;

    // for displaying detail of the model attributes
    const [modelValue, setModelValue] = useState(null);
    const [loading, setLoading] = useState(true);

    const handleRecievedModelVal = useCallback((data) => {
        setLoading(false);
        // args are rendered on parent component
        if (data !== 404 && data[0] !== "args") {
            setModelValue(data[1]);
        }
    });

    useEffect(() => {
        socket.emit("get_model_value_by_id_n_key", batchId, modelKey);
    }, []); // component did mount


    const handleCloseModal = () => {
        setModalOpen(false);
        setModelValue(null);
        setModelKey(null);
        setLoading(true);
    }

    useEffect(() => {
        socket.on("get_model_value_by_id_n_key", (data) => handleRecievedModelVal(data));
    }, [socket, handleRecievedModelVal]);

    return <Modal
        title={`Detail for ${modelKey}`}
        visible={modalOpen}
        footer={null}
        destroyOnClose={true}
        centered
        width={1200}
        onCancel={() => handleCloseModal()}>
        <div style={{
            overflow: "auto",
            height: "50vh",
        }}>
            {loading ? <Spin /> : <div>{modelValue}</div>}
        </div>
    </Modal>;
}

export default ModelAtrributeModal;