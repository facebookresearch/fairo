/*
Copyright (c) Facebook, Inc. and its affiliates.

The content for model visualization for a pipeline.

The **pipeline** can be nlu/tao or vision; the pipeline information was reterived from path parameter so no props are required.

Usage:
<PipelineModelVizContent />
*/

import { SyncOutlined } from "@ant-design/icons";
import { Progress, Spin, Tag, Typography } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { SocketContext } from "../../../context/socket";
import { ViewLossAccCard } from "../detail/asset/modelCard";

const contentDivStyle = { padding: "0 24px 0 24px" };
const paddingBottomSytle = { padding: "0 0 12px 0" };
const { CheckableTag } = Tag;

const PipelineModelVizContent = (props) => {
    const location = useLocation();
    const socket = useContext(SocketContext);
    const [modelBids, setModelBids] = useState(null);
    const [selectedBids, setSelectedBids] = useState(null);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [modelDict, setModelDict] = useState({
        loss: {},
        acc: {}
    });

    const handleReceivedModelBids = (data) => {
        setModelBids(data);
        setSelectedBids(data);
    }

    const handleReceivedModelLossAcc = (data) => {
        if (data !== 404 && modelBids && modelBids !== 404 && !(data[1] in modelDict.loss)) {
            setModelDict((prevModelDict) => ({
                ...prevModelDict,
                loss: {...prevModelDict.loss, [data[1]]: data[0].loss},
                acc: {...prevModelDict.acc, [data[1]]: data[0].acc},
            }));
            setCurrentIdx(currentIdx + 1);
        }
    }

    const handleToogleCheckedBid = (bid, checked) => {
        // toggle checked for the selected bid
        const nextSelectedBids = checked
            ? [...selectedBids, bid]
            : selectedBids.filter((b) => b !== bid);
        setSelectedBids(nextSelectedBids);
    }

    useEffect(() => {
        socket.on("get_best_model_bids_by_pipeline", (data) => handleReceivedModelBids(data));
        socket.on("get_best_model_loss_acc_by_id", (data) => handleReceivedModelLossAcc(data));
    }, [socket, handleReceivedModelBids, handleReceivedModelLossAcc]);

    useEffect(() => {
        socket.emit("get_best_model_bids_by_pipeline", location.pathname.substring(1));
    }, []);

    useEffect(() => {
        if (modelBids && modelBids !== 404 && currentIdx >= 0 && currentIdx < modelBids.length) {
            socket.emit("get_best_model_loss_acc_by_id", modelBids[currentIdx]);
        }
    }, [modelBids, currentIdx]);

    return <div>
        <Typography.Title level={4}>{"View Model Accuracy & Loss"}</Typography.Title>
        {
            (!modelBids || !selectedBids) && <Spin />
        }
        {
            modelBids && modelBids !== 404 && selectedBids && selectedBids !== 404
            &&
            <div
                style={paddingBottomSytle}
            >
                Showing {modelBids.map((bid) =>
                    <CheckableTag
                        key={bid}
                        checked={selectedBids.includes(bid)}
                        onChange={(checked) => handleToogleCheckedBid(bid, checked)}
                    >
                        {bid}
                    </CheckableTag>
                )}
            </div>
        }
        <div style={contentDivStyle}>
            {
                // show progress bar after getting model batch ids
                modelBids && modelBids !== 404 && currentIdx !== modelBids.length &&
                <div>
                    <span><SyncOutlined spin /> Loading Model Data... </span>
                    <Progress percent={Math.trunc((currentIdx) / modelBids.length * 100)} />
                </div>
            }

            {
                // show graph when loading is done
                modelBids && modelBids !== 404 && currentIdx === modelBids.length &&
                <div>
                    <ViewLossAccCard data={modelDict} width={1200} height={800} bids={selectedBids} />
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
