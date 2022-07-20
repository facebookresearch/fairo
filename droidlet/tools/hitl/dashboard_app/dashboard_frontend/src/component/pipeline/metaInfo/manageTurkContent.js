import { Spin, Tabs, Typography } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../context/socket";
import { snakecaseToWhitespaceSep, toFirstCapital } from "../../../utils/textUtils";
import TurkList from "./turkList";

const { TabPane } = Tabs;

const ManageTurkContent = (props) => {
    const pipelineType = props.pipelineType;

    const socket = useContext(SocketContext);
    const [turkData, setTurkData] = useState(null);



    const handleRecivedTurkList = (data) => {
        const processedData = {};

        for (let idx = 0; idx < Object.keys(data).length; idx++) {
            const k = Object.keys(data)[idx];

            const allowList = Array.from(new Set(data[k]["allow"])); // used set for dedup
            const blockSet = new Set(data[k]["block"]);

            const softblockSet = new Set(data[k]["softblock"]);

            processedData[k] = allowList.map((tid) => (
                {
                    "id": tid,
                    "status":
                        blockSet.has(tid) ?
                            "block" :
                            (
                                softblockSet.has(tid) ?
                                    "softblock" :
                                    "allow"
                            )
                }
            ));
        }

        setTurkData(processedData);
    }

    const getTurkList = () => {
        socket.emit("get_turk_list_by_pipeline", pipelineType.label);
    }

    useEffect(() => { !turkData && getTurkList() }, []); // component did mount

    useEffect(() => {
        socket.on("get_turk_list_by_pipeline", (data) => handleRecivedTurkList(data));
    }, [socket, handleRecivedTurkList]);

    useEffect(() => { }, turkData); // update on state change

    return <div style={{ textAlign: "left" }}>
        <Typography.Title level={5}>Manage Turk List</Typography.Title>
        {
            turkData ?
                <div style={{ paddingRight: "16px" }}>
                    <Tabs defaultActiveKey={Object.keys(turkData)[0]}>
                        {Object.entries(turkData).map(([name, data]) =>
                            <TabPane tab={snakecaseToWhitespaceSep(name)} key={name}>
                                <TurkList turkListName={name} turkListData={data} />
                            </TabPane>)}
                    </Tabs>
                </div>
                :
                <Spin />
        }
    </div>
}

export default ManageTurkContent;