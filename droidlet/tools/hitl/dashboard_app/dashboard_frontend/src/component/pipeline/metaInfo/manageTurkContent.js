import { Spin, Typography } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../context/socket";
import TurkList from "./turkList";

const ManageTurkContent = (props) => {
    const pipelineType = props.pipelineType;

    const socket = useContext(SocketContext);
    const [turkData, setTurkData] = useState(null);

    const handleRecivedTurkList = (data) => {
        const processedData = {};

        for (let idx = 0; idx < Object.keys(data).length; idx++) {
            const k = Object.keys(data)[idx];
            const allowList = data[k]["allow"].map((o) => ({ 'id': o, 'blocked': false }));
            const blockList = data[k]["block"].map((o) => ({ 'id': o, 'blocked': true }));
            const softblockList = data[k]["softblock"];
            processedData[k] = allowList.concat(blockList);
        }
        setTurkData(processedData);
    }

    const getTurkList = () => {
        socket.emit("get_turk_list_by_pipeline", pipelineType.label);
    }

    useEffect(() => getTurkList(), []); // component did mount

    useEffect(() => {
        socket.on("get_turk_list_by_pipeline", (data) => handleRecivedTurkList(data));
    }, [socket, handleRecivedTurkList]);

    useEffect(() => { }, turkData); // update on state change

    return <div style={{ textAlign: 'left' }}>
        <Typography.Title level={5}>Manage Turk List</Typography.Title>
        {
            turkData ?
                <div style={{paddingRight: '16px'}}>
                    {Object.entries(turkData).map(([name, data]) => <TurkList turkListName={name} turkListData={data} />)}
                </div>
                :
                <Spin />
        }
    </div>
}

export default ManageTurkContent;