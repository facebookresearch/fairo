import { Button, Checkbox, Table } from "antd";
import React, { useEffect, useState } from "react";
import { LockOutlined, UnlockOutlined } from '@ant-design/icons';


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
    const [editable, setEditable] = useState(false); 

    const handleOnClickCheckbox = (checked, id) => {
        // TODO: end request to backend, update only when backend returns success
        const newDataList = listData.map((o) => (o.id === id ? {'id': o.id, 'blocked': checked}: {'id': o.id, 'blocked': o.blocked}));
        setListData(newDataList);
    }

    useEffect(() => {}, [listData, editable]);

    return <div>
        <div style={{textAlign: 'left', paddingBottom: '12px'}}>
            <Button 
                type="primary" 
                icon={editable ? <UnlockOutlined />: <LockOutlined />} 
                onClick={() => setEditable(!editable)}>
                    {editable ? "Lock": "Unlock"} Editing
            </Button>
        </div> 
        <Table
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
            dataSource={listData}
        />
    </div>;
};

export default TurkList;