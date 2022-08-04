/*
Copyright (c) Facebook, Inc. and its affiliates.

The content for model visualization for a pipeline.

The **pipeline** can be nlu/tao or vision; the pipeline information was reterived from path parameter so no props are required.

Usage:
<PipelineModelVizContent />
*/

import { SyncOutlined } from "@ant-design/icons";
import { Progress, Typography } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { SocketContext } from "../../../context/socket";
import { ViewLossAccCard } from "../detail/asset/modelCard";

const PipelineModelVizContent = (props) => {
    const location = useLocation();
    const socket = useContext(SocketContext);
    const [modelBids, setModelBids] = useState(null);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [modelDict, ] = useState({});
    const [modelCombined, setModelCombined] = useState(null);

    const handleReceivedModelBids = (data) => {
        setModelBids(data.map((bid) => parseInt(bid)));
    }

    const handleReceivedModelLossAcc = (data) => {
        if (data !== 404) {
            modelDict[data[1]] = data[0];

            if (modelBids && modelBids !== 404 && Object.keys(modelDict).length < modelBids.length) {
                setCurrentIdx(Object.keys(modelDict).length);
            } else if (modelBids && modelBids !== 404 && Object.keys(modelDict).length === modelBids.length) {
                // finished loading, combine based on bid sequence
                const modelArr = modelBids.map((bid) => (modelDict[bid]));
                let comb = {}

                for (let i = 0; i < modelArr.length; i++) {
                    const o = modelArr[i];
                    for (let j = 0; j < Object.keys(o).length; j++) {
                        const key = Object.keys(o)[j];
                        if (!(key in comb)) {
                            comb[key] = o[key];
                        } else {
                            comb[key] = [...comb[key], ...o[key]]
                        }
                    }
                }
                setModelCombined(comb);
            }
        }
    }

    useEffect(() => {
        socket.on("get_best_model_bids_by_pipeline", (data) => handleReceivedModelBids(data));
        socket.on("get_best_model_loss_acc_by_id", (data) => handleReceivedModelLossAcc(data));
    }, [socket, handleReceivedModelBids, handleReceivedModelLossAcc]);

    useEffect(() => {
        socket.emit("get_best_model_bids_by_pipeline", location.pathname.substring(1));
    }, []);

    useEffect(() => {
        if (modelBids && modelBids !== 404 && currentIdx >= 0) {
            socket.emit("get_best_model_loss_acc_by_id", modelBids[currentIdx]);
        }
    }, [modelBids, currentIdx]);

    return <div>
        <Typography.Title level={4}>{"View Model Accuracy & Loss"}</Typography.Title>
        <div style={{ padding: "0 24px 0 24px" }}>
            {
                // show progress bar after getting model batch ids
                modelBids && modelBids !== 404 && (Object.keys(modelDict).length) !== modelBids.length &&
                <div> <span><SyncOutlined spin /> Loading Model Data... </span>
                    <Progress percent={Math.trunc((Object.keys(modelDict).length) / modelBids.length * 100)} />
                </div>
            }

            {
                // show graph when loading is done
                modelBids && modelBids !== 404 && (Object.keys(modelDict).length) === modelBids.length && modelCombined &&
                <div>
                    <ViewLossAccCard data={modelCombined} width={1500}/>
                </div>
            } 
            {
                // show no data when not having any batch ids
                modelBids && modelBids === 404 && 
                <div>
                    Sorry, no model data is avaible yet. 
                </div>
            }
        </div>

    </div>
}

export default PipelineModelVizContent;