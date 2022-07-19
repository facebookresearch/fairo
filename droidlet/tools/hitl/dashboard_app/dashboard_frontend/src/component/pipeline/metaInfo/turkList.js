/*
Copyright (c) Facebook, Inc. and its affiliates.

Turk list used for viewing & setting turks to block or not blocked.

Usage:
<TurkList pipelineType={pipelineType} />
*/
import { Button, Checkbox, Collapse, Input, Table, Typography } from "antd";
import React, { useEffect, useState } from "react";
import { LockOutlined, UnlockOutlined } from '@ant-design/icons';

const { Search } = Input;

const TurkList = (props) => {
    const turkListName = props.turkListName;
    const [listData, setListData] = useState(props.turkListData);
    const [displayData, setDisplayData] = useState(props.turkListData);

    const [searchKey, setSearchKey] = useState(null);
    const [editable, setEditable] = useState(false);

    const handleOnClickCheckbox = (checked, id) => {
        // TODO: end request to backend, update only when backend returns success
        const newDataList = listData.map((o) => (o.id === id ? { 'id': o.id, 'blocked': checked } : { 'id': o.id, 'blocked': o.blocked }));
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
                title: 'Blocked',
                dataIndex: 'blocked',
                sorter: (one, other) => (one.blocked === other.blocked ? 0 : (one.blocked < other.blocked ? -1 : 1)),
                render: (blocked, row) =>
                    <Checkbox onChange={(e) => handleOnClickCheckbox(e.target.checked, row.id)}
                        disabled={!editable}
                        checked={blocked} />
            }]}
            dataSource={displayData}
        />
    </div>;
};

export default TurkList;