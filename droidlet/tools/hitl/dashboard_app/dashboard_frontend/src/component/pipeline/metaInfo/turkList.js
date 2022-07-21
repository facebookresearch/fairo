/*
Copyright (c) Facebook, Inc. and its affiliates.

Turk list used for viewing & setting turks to block or not blocked.

Usage:
<TurkList taskType={taskType} turkListData={turkListData} />
*/
import { Alert, Button, Input, Radio, Spin, Table } from "antd";
import React, { useContext, useEffect, useState } from "react";
import { LockOutlined, UnlockOutlined } from "@ant-design/icons";
import { reteriveLabelByValue } from "../../../utils/textUtils";
import { SocketContext } from "../../../context/socket";

const { Search } = Input;
const STATUS_TYPES = [
    { "label": "Allow", "value": "allow" },
    { "label": "Pilotblock", "value": "block" },
    { "label": "Block", "value": "softblock" },
]

const TurkList = (props) => {
    const socket = useContext(SocketContext);
    const taskType = props.taskType;
    const [listData, setListData] = useState(props.turkListData);
    const [displayData, setDisplayData] = useState(props.turkListData);
    const [tmpData, setTmpData] = useState(props.turkListData);
    const [idInUpdating, setIdInUpdating] = useState(null);
    const [searchKey, setSearchKey] = useState(null);
    const [editable, setEditable] = useState(false);

    const [showFailure, setShowFailure] = useState(false);

    const handleRecievedTurkUpdateRes = (data) => {
        if (data === 200) {
            // update success
            setListData(tmpData);
            setShowFailure(false);
            setIdInUpdating(null);
            setDisplayData(searchKey ? tmpData.filter((o) => (String(o.id).includes(searchKey))) : tmpData);
        } else {
            // failed to update
            setShowFailure(true);
            setDisplayData(searchKey ? listData.filter((o) => (String(o.id).includes(searchKey))) : listData);
        }
    }

    const handleRadioSel = (value, id, previousStatus) => {
        setIdInUpdating(id);
        socket.emit("update_turk_qual_by_tid", id, taskType, value, previousStatus);
        // save a temp data
        const newDataList = listData.map((o) => (o.id === id ? { "id": o.id, "status": value } : { "id": o.id, "status": o.status }));
        setTmpData(newDataList);
    }

    useEffect(() => {
        socket.on("update_turk_qual_by_tid", (data) => handleRecievedTurkUpdateRes(data));
    }, [socket, handleRecievedTurkUpdateRes]);

    const onSearch = (searchBoxValue) => {
        if (searchBoxValue) {
            setDisplayData(listData.filter((o) => (String(o.id).includes(searchBoxValue.toUpperCase()))));
            setSearchKey(searchBoxValue.toUpperCase());
        } else {
            setDisplayData(listData);
            setSearchKey(null);
        }
    }

    useEffect(() => { }, [displayData, idInUpdating]);

    const handleCloseAlert = () => {
        setIdInUpdating(null);
        setShowFailure(false);
    }

    return <div>
        <div style={{ extAlign: "left", paddingBottom: "12px" }}>
            <Search
                placeholder="Search by Id"
                style={{ width: "30%" }}
                allowClear
                onSearch={onSearch}
                enterButton />
            <div style={{ float: "right", paddingRight: "18px", display: "inline-block" }}>
                <Button
                    type="primary"
                    icon={editable ? <UnlockOutlined /> : <LockOutlined />}
                    onClick={() => setEditable(!editable)}>
                    {editable ? "Lock" : "Unlock"} Editing
                </Button>
            </div>
        </div>
        <Table
            style={{ paddingRight: "24px" }}
            columns={[{
                title: "Id",
                dataIndex: "id",
                sorter: (one, other) => (one.id === other.id ? 0 : (one.id < other.id ? -1 : 1)),
            }, {
                title: editable ? "Edit tatus" : "Status",
                dataIndex: "status",
                filters: STATUS_TYPES.map((t) => ({ "text": t.label, "value": t.value })),
                onFilter: (val, row) => (row.status === val),
                sorter: (one, other) => (one.status.localeCompare(other.status)),
                render: (status, row) =>
                    row.id === idInUpdating ?
                        (showFailure ? 
                        <Alert 
                            description="Update failed, please try again later."
                            type="error"
                            closable
                            onClose={handleCloseAlert}
                        />
                        :
                        <div><Spin style={{ marginRight: '12px' }} />Updating Status</div>) :
                        (editable ?
                            <Radio.Group onChange={(e) => handleRadioSel(e.target.value, row.id, status)} value={status}>
                                {
                                    STATUS_TYPES.map((t) =>
                                        <Radio value={t.value}>{t.label}</Radio>
                                    )
                                }
                            </Radio.Group> :
                            <div>
                                {reteriveLabelByValue(status, STATUS_TYPES)}
                            </div>)

            }]}
            dataSource={displayData}
        />
    </div>;
};

export default TurkList;