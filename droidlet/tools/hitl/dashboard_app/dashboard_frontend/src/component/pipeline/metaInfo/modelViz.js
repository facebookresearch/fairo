import React, { useContext, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { SocketContext } from "../../../context/socket";

const PipelineModelVizContent = (props) => {
    const location = useLocation();
    const socket = useContext(SocketContext);
    const [modelBids, setModelBids] = useState(null);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [modelDict, setModelDict] = useState({});
    const [loadingPrecent, setPercent] = useState(0);
    
    const handleReceivedModelBids = (data) => {
        console.log(data)
        setModelBids(data.map((bid) => parseInt(bid)));
    }

    const handleReceivedModelLossAcc = (data) => {
        // data = JSON.parse(data);
        console.log(data);

        if (data !== 404) {
            modelDict[data[1]] = data[0];
            console.log(modelDict)
            setPercent(Object.keys(modelDict).length);
            console.log(Object.keys(modelDict).length);

            if (modelBids && modelBids !== 404 && currentIdx >= 0 && currentIdx + 1 < modelBids.length) {
                setCurrentIdx(currentIdx + 1);
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
        Model
        {loadingPrecent}
    </div>
}

export default PipelineModelVizContent;