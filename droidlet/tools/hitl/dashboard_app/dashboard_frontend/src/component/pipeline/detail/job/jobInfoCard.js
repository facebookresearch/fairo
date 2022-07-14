/*
Copyright (c) Facebook, Inc. and its affiliates.

Detail info of a job. 
*/
import { Button, Card, Descriptions, List, Modal, Spin, Typography } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import { Link, useOutletContext, useParams } from "react-router-dom";
import { JOB_STATUS_CONSTANTS, JOB_STATUS_ORDER } from "../../../../constants/runContants";
import { SocketContext } from "../../../../context/socket";
import { toFirstCapital } from "../../../../utils/textUtils";
import LogDisplayCmp from "../../../common/logDisplay";

const JobInfoCard = (props) => {
    const batchId = useParams().batch_id;
    const job = useParams().job;
    const [sessionList, setSessionList] = useState([]);
    const [modalOpen, setModalOpen] = useState(false);
    const [currentSession, setCurrentSession] = useState(null);
    const [sessionLog, setSessionLog] = useState(null);

    const socket = useContext(SocketContext);

    let jobInfo = Object.entries(useOutletContext().metaInfo)
        .filter((o) =>
            o[0].toLowerCase() === job)[0][1];
    jobInfo = Object.entries(jobInfo)
        .sort((one, another) =>
            (JOB_STATUS_ORDER.indexOf(one[0]) - JOB_STATUS_ORDER.indexOf(another[0])));

    const handleRecievedSessionist = useCallback((data) => {
        setSessionList(data);
    }, []);

    const handleRecivedSessionLog = useCallback((data) => {
        setSessionLog(data);
    }, []);

    useEffect(() => {
        socket.on("get_interaction_sessions_by_id", (data) => handleRecievedSessionist(data));
        socket.on("get_interaction_session_log", (data) => handleRecivedSessionLog(data));
    }, [socket, handleRecievedSessionist, handleRecivedSessionLog])

    const getDesciptionText = (o) => {
        if (!o[1]) {
            // not available yet
            return "NA";
        } else if (o[0] === "DASHBOARD_VER") {
            // dashboard version, need to remove sha256 prefix
            const idx = o[1].indexOf("sha256:");
            return o[1].substring(idx + "sha256.".length);
        } else if (o[0].includes("_TIME")) {
            // time: need to remove after ss
            const t_idx = o[0] === "START_TIME" ? 0 : o[1].length - 1;
            const idx = o[1][t_idx].indexOf(".");
            return o[1][t_idx].substring(0, idx);
        } else if (typeof (o[1]) === "boolean") {
            // boolean fields
            return o[1] ? "Yes" : "No";
        }
        // get session log if has the session log 
        if (o[0] === "NUM_SESSION_LOG" && sessionList.length === 0) {
            socket.emit("get_interaction_sessions_by_id", batchId);
        }

        return typeof (o[1]) === "string" ? toFirstCapital(o[1]) : o[1];
    }

    useEffect(() => {
        if (currentSession) {
            const id_info_obj = {
                "batch_id": batchId,
                "session_id": currentSession
            }
            socket.emit("get_interaction_session_log", JSON.stringify(id_info_obj));
        }
    }, [currentSession]);

    const handleViewSession = (session) => {
        setModalOpen(true);
        setCurrentSession(session);
    }

    const handleCloseSessionModal = () => {
        setModalOpen(false);
        setCurrentSession(null);
        setSessionLog(null);
    }

    return <div style={{ 'paddingLeft': '12px' }}>
        <Card
            title={`${toFirstCapital(job)} Jobs`}
            extra={<Button type="link"><Link to="../">Close</Link></Button>}
        >
            <Descriptions column={4} bordered>
                {jobInfo.map(
                    (o) => (
                        <Descriptions.Item label={JOB_STATUS_CONSTANTS[o[0]].label} span={JOB_STATUS_CONSTANTS[o[0]].span}>
                            {getDesciptionText(o)}
                        </Descriptions.Item>
                    )
                )}
            </Descriptions>
            {
                // view session list if there is a session list and the job is interaction
                job === "interaction" && (sessionList.length ?
                    <div
                        style={{
                            paddingTop: "12px",
                            overflow: "auto",
                            textAlign: "left",
                            height: 600,
                        }}
                    >
                        <List
                            header={<Typography.Text strong>Sessions</Typography.Text>}
                            size="small"
                            bordered
                            dataSource={sessionList}
                            renderItem={item => (
                                <List.Item>
                                    <Typography.Text>{item}</Typography.Text>
                                    <Button type="link" onClick={() => handleViewSession(item)}>View Session Log</Button>
                                </List.Item>
                            )}
                        />
                    </div>
                    : <Spin />)
            }
        </Card>
        {/* session modal starts */}
        <Modal
            title={`Session Log of ${currentSession}`}
            visible={modalOpen}
            footer={null}
            destroyOnClose={true}
            centered
            width={1200}
            onCancel={() => handleCloseSessionModal()}
        >
            <div
                style={{
                    overflow: "auto",
                    height: "50vh",
                }}
            >
                {sessionLog ? <LogDisplayCmp log={sessionLog}/> : <Spin />}
            </div>
        </Modal>
        {/* session modal ends */}
    </div>;
}
export default JobInfoCard;