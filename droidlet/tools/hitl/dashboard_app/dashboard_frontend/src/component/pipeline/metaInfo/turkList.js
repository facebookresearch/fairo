import { Button, Checkbox, Input, Table, Typography } from "antd";
import React, { useEffect, useState } from "react";
import { LockOutlined, UnlockOutlined } from '@ant-design/icons';

const { Search } = Input;

const fakeTurkList = [
    {
        'id': 1213023,
        'blocked': true
    },
    {
        'id': 32349712,
        'blocked': false
    }, {
        'id': 143412,
        'blocked': false
    }, {
        'id': 435304,
        'blocked': false
    }, {
        'id': 2234948,
        'blocked': true
    }, {
        'id': 9453311,
        'blocked': false
    }, {
        'id': 134524441,
        'blocked': false
    },
]

const TurkList = (props) => {
    const [listData, setListData] = useState(fakeTurkList); // TODO: use turk list from backend
    const [searchKey, setSearchKey] = useState(null);
    const [displayData, setDisplayData] = useState(fakeTurkList);
    const [editable, setEditable] = useState(false);

    const handleOnClickCheckbox = (checked, id) => {
        // TODO: end request to backend, update only when backend returns success
        const newDataList = listData.map((o) => (o.id === id ? { 'id': o.id, 'blocked': checked } : { 'id': o.id, 'blocked': o.blocked }));
        setListData(newDataList);
        setDisplayData(searchKey ? newDataList.filter((o) => (String(o.id).includes(searchKey))): newDataList);
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

    useEffect(() => {}, [displayData]);

    return <div style={{ textAlign: 'left' }}>
        <Typography.Title level={5}>Manage Turk List</Typography.Title>
        <div style={{ paddingBottom: '12px'}}>
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
            style={{paddingRight: '24px'}}
            columns={[{
                title: 'Id',
                dataIndex: 'id',
                sorter: (one, other) => (one === other ? 0 : (one < other ? -1 : 1)),
            }, {
                title: 'Blocked',
                dataIndex: 'blocked',
                sorter: (one, other) => (one === other ? 0 : (one < other ? -1 : 1)),
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