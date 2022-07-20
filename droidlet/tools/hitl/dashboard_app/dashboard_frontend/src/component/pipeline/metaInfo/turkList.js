/*
Copyright (c) Facebook, Inc. and its affiliates.

Turk list used for viewing & setting turks to block or not blocked.

Usage:
<TurkList turkListName={turkListName} turkListData={turkListData} />
*/
import { Button, Input, Radio, Table } from "antd";
import React, { useEffect, useState } from "react";
import { LockOutlined, UnlockOutlined } from '@ant-design/icons';
import { toFirstCapital } from "../../../utils/textUtils";

const { Search } = Input;

const TurkList = (props) => {
    const turkListName = props.turkListName;
    const [listData, setListData] = useState(props.turkListData);
    const [displayData, setDisplayData] = useState(props.turkListData);

    const [searchKey, setSearchKey] = useState(null);
    const [editable, setEditable] = useState(false);

    const handleRadioSel = (value, id) => {
        // TODO: end request to backend, update only when backend returns success
        const newDataList = listData.map((o) => (o.id === id ? { 'id': o.id, 'status': value } : { 'id': o.id, 'status': o.status }));
        setListData(newDataList);
        setDisplayData(searchKey ? newDataList.filter((o) => (String(o.id).includes(searchKey))) : newDataList);
    }

    const onSearch = (searchBoxValue) => {
        if (searchBoxValue) {
            setDisplayData(listData.filter((o) => (String(o.id).includes(searchBoxValue))));
            setSearchKey(searchBoxValue);
        } else {
            setDisplayData(listData);
            setSearchKey(null);
        }
    }

    useEffect(() => { }, [displayData]);

    return <div>
        <div style={{ extAlign: 'left', paddingBottom: '12px' }}>
            <Search
                placeholder="Search by Id"
                style={{ width: '30%' }}
                allowClear
                onSearch={onSearch}
                enterButton />
            <div style={{ float: 'right', paddingRight: '18px', display: 'inline-block' }}>
                <Button
                    type="primary"
                    icon={editable ? <UnlockOutlined /> : <LockOutlined />}
                    onClick={() => setEditable(!editable)}>
                    {editable ? "Lock" : "Unlock"} Editing
                </Button>
            </div>
        </div>
        <Table
            style={{ paddingRight: '24px' }}
            columns={[{
                title: 'Id',
                dataIndex: 'id',
                sorter: (one, other) => (one.id === other.id ? 0 : (one.id < other.id ? -1 : 1)),
            }, {
                title: editable ? 'Edit tatus' : 'Status',
                dataIndex: 'status',
                sorter: (one, other) => (one.status.localeCompare(other.status)),
                render: (status, row) =>
                    editable ?
                        <Radio.Group onChange={(e) => handleRadioSel(e.target.value, row.id)} value={status}>
                            <Radio value={"allow"}>Allow</Radio>
                            <Radio value={"block"}>Block</Radio>
                            <Radio value={"softblock"}>Softblock</Radio>
                        </Radio.Group> :
                        <div>
                            {toFirstCapital(status)}
                        </div>

            }]}
            dataSource={displayData}
        />
    </div>;
};

export default TurkList;