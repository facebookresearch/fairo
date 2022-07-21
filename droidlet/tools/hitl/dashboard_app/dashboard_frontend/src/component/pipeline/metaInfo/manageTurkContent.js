/*
Copyright (c) Facebook, Inc. and its affiliates.

Wrapper for turk list, reterive turk list data and render the TurkList component.

Usage:
<ManageTurkContent pipelineType={pipelineType} />
*/
import { CloudSyncOutlined, SyncOutlined } from "@ant-design/icons";
import { Button, Spin, Tabs, Typography } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../context/socket";
import { snakecaseToWhitespaceSep } from "../../../utils/textUtils";
import TurkList from "./turkList";

const { TabPane } = Tabs;

const ManageTurkContent = (props) => {
    const pipelineType = props.pipelineType;
    const socket = useContext(SocketContext);
    const [turkData, setTurkData] = useState(null);
    const [syncing, setSyncing] = useState(false);

    const handleRecivedTurkList = (data) => {
        const processedData = {};

        for (let idx = 0; idx < Object.keys(data).length; idx++) {
            const k = Object.keys(data)[idx];

            // get blocklist (passed pilot test)
            // anyone must be on allowlist too if on block list, so only checking block list & softblocklist
            const blockList = Array.from(new Set(data[k]["block"])); // used set for dedup
            const softblockSet = new Set(data[k]["softblock"]);

            processedData[k] = blockList.map((tid) => (
                {
                    "id": tid,
                    "status":
                        softblockSet.has(tid) ?
                            "softblock" :
                            "block"
                }
            ));
        }

        setTurkData(processedData);
    };

    const getTurkList = () => {
        socket.emit("get_turk_list_by_pipeline", pipelineType.label);
    };

    const handleRecivedSyncedData = (data) => {
        if (data === 200) {
            setSyncing(false);
            getTurkList();
        } else {
            setSyncing(false);
            alert("Update failed");
            getTurkList();
        }
    };

    const handleSync = () => {
        socket.emit("update_local_turk_ls_to_sync");
        setTurkData(null);
        setSyncing(true);
    };

    useEffect(() => { 
        getTurkList();
    }, []); // component did mount

    useEffect(() => {
        socket.on("get_turk_list_by_pipeline", (data) => handleRecivedTurkList(data));
        socket.on("update_local_turk_ls_to_sync", (data) => handleRecivedSyncedData(data));
    }, [socket, handleRecivedTurkList, handleRecivedSyncedData]);

    useEffect(() => {}, turkData); // update on state change

    return <div style={{ textAlign: "left" }}>
        <Typography.Title level={5}>Manage Turk List</Typography.Title>
        <div>
            Showing mephisto local data. 
            <Button 
                onClick={handleSync} 
                icon={syncing?  <SyncOutlined spin/> : <CloudSyncOutlined />} 
                type="primary" 
                size="small" 
                disabled={syncing}
                style={{marginLeft: '12px'}}>
                    {syncing? "Syncing..." : "Sync with S3"}
            </Button>
        </div>
        {
            turkData ?
                <div style={{ paddingRight: "16px" }}>
                    <Tabs defaultActiveKey={Object.keys(turkData)[0]}>
                        {Object.entries(turkData).map(([name, data]) =>
                            <TabPane tab={snakecaseToWhitespaceSep(name)} key={name}>
                                <TurkList taskType={name} turkListData={data} />
                            </TabPane>)}
                    </Tabs>
                </div>
                :
                <Spin />
        }
    </div>
}

export default ManageTurkContent;